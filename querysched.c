#include "postgres.h"
#include "storage/lwlock.h"
#include "utils/elog.h"
#include "stdlib.h"
#include "miscadmin.h"
#include "parser/parsetree.h"
#include "access/xact.h"
#include "nodes/plannodes.h"
#include "utils/builtins.h"
#include "storage/lmgr.h"
#include "nodes/nodes.h"

#include "pg_ivm.h"
#include <netinet/in.h>
#include <resolv.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

#pragma pack(1)

typedef struct payload_t {
    double embedding[1024];
} payload;

typedef struct response_t {
    uint32_t decision;
} response;

typedef struct feedback_t {
		double reward;
} feedback;

#pragma pack()

extern int decision_server_socket;
extern int getIndexForTableEmbeeding(Oid oid);
void RescheduleUseMinTableAffected(HTAB *queryTable, ScheduleState *state);
void RescheduleUseFCFS(HTAB *queryTable, ScheduleState *state);
void RescheduleUseHotTableFirst(HTAB *queryTable, ScheduleState *state);
void RescheduleWithServer(HTAB *queryTable, ScheduleState *state);
int embed_plan_node(double *tensor, Plan *plan, int start);
void sendMsg(int sock, void* msg, uint32_t msgsize);
void sendRequest(QueryTableEntry *qte);
uint32 recvDecision();

void sendMsg(int sock, void* msg, uint32_t msgsize)
{
    if (write(sock, msg, msgsize) < 0)
    {
				ereport(ERROR, (errcode(ERRCODE_IO_ERROR), errmsg("Can't send message.")));
        close(sock);
    }
    return;
}

void sendRequest(QueryTableEntry *qte)
{
	sendMsg(decision_server_socket, qte->embedding, sizeof(payload));
}

uint32
recvDecision()
{
	uint32_t decision;
	ssize_t nread;

	nread = read(decision_server_socket, &decision, sizeof(uint32_t));
	if (nread == 0){
		elog(ERROR, "Connection closed by server.");
		close(decision_server_socket);
	}
	else if (nread < 0){
		elog(ERROR, "Error reading from server.");
		close(decision_server_socket);
	}
	else if (nread != sizeof(uint32_t))
	{
		ereport(ERROR, (errcode(ERRCODE_IO_ERROR), errmsg("Can't receive the decision from server.")));
		close(decision_server_socket);
	}
	elog(IVM_LOG_LEVEL, "Received decision: %d", decision);

	return decision;
}

typedef struct TableRef
{
	Oid oid;
	int num_of_ref;
} TableRef;

QueryTableEntry *
LogQuery(HTAB *queryTable, ScheduleState *state, PlannedStmt *plannedStmt, const char *query_string)
{
	int oidIndex;
	bool found;
	QueryTableEntry *query_entry;
	ListCell *roid;
	QueryTableKey key;
	int effective_index = 0;
	double *tensor;
	int table_one_hot_index = 0;

	memset(&key, 0, sizeof(QueryTableKey));
	key.pid = MyProcPid;
	strcpy(key.query_string, query_string);

	if (state->querynum >= MAX_QUERY_NUM)
		ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("Too many queries in the system")));

	state->querynum++;

	query_entry = (QueryTableEntry *) hash_search(queryTable, &key, HASH_ENTER, &found);

	if (found)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DUPLICATE_OBJECT),
				 errmsg("Found duplicate entry key, query_string: %s, previous query_string: %s",
						query_string,
						query_entry->key.query_string)));
	}

	memset(query_entry, 0, sizeof(QueryTableEntry));
	strcpy(query_entry->key.query_string, query_string);
	query_entry->key.pid = MyProcPid;
	query_entry->status = QUERY_BLOCKED;
	query_entry->xid = GetCurrentTransactionId();

	elog(IVM_LOG_LEVEL, "Logging Transactionid: %u", query_entry->xid);

	tensor = query_entry->embedding;

	tensor[0] = plannedStmt->commandType;
	tensor[1] = plannedStmt->hasReturning;
	tensor[2] = plannedStmt->hasModifyingCTE;
	tensor[3] = plannedStmt->canSetTag;
	tensor[4] = plannedStmt->transientPlan;
	tensor[5] = plannedStmt->dependsOnRole;
	tensor[6] = plannedStmt->parallelModeNeeded;

	/* Not sure if this is correct.*/
	oidIndex = 0;
	foreach (roid, plannedStmt->relationOids)
	{
		if (oidIndex > MAX_AFFECTED_TABLE)
		{
			elog(ERROR, "Too many affected tables for query: %s", query_string);
		}
		query_entry->affected_tables[oidIndex++] = lfirst_oid(roid);

		table_one_hot_index = getIndexForTableEmbeeding(lfirst_oid(roid));

		if (table_one_hot_index == -1){
			elog(WARNING, "Cannot find table embedding for table oid: %d", lfirst_oid(roid));
			continue;
		}

		tensor[getIndexForTableEmbeeding(lfirst_oid(roid)) + 7] = 1;
	}

	effective_index = embed_plan_node(tensor, plannedStmt->planTree, 15);
	memset(&tensor[effective_index], -1, (QUERY_EMBEDDING_SIZE - effective_index) * sizeof(double));
	return query_entry;
}

void
RemoveLoggedQuery(QueryDesc *queryDesc, HTAB *queryHashTable, ScheduleState *schedule_state)
{
	QueryTableEntry *query;
	bool found;
	QueryTableKey key;

	memset(&key, 0, sizeof(QueryTableKey));
	key.pid = MyProcPid;
	strcpy(key.query_string, queryDesc->sourceText);

	query = hash_search(queryHashTable, &key, HASH_REMOVE, &found);

	if (!found || query == NULL)
	{
		elog(IVM_LOG_LEVEL, "Pid:%d: Cannot find Query:%s", MyProcPid, queryDesc->sourceText);
		return;
	}

	elog(IVM_LOG_LEVEL, "Removing Query xid:%d", query->xid);
	schedule_state->querynum--;
}
void
RescheduleWithServer(HTAB *queryTable, ScheduleState *state)
{
	HASH_SEQ_STATUS status;
	QueryTableEntry *query_entry;
	int available_num;
	uint32_t decision = 0;
	List *listing = NIL;
	ListCell *curr = NULL;
	static double endTensor[QUERY_EMBEDDING_SIZE];
	int giving_up = 0;

	available_num = MAX_CONCURRENT_QUERY - state->runningQuery;

	if (available_num <= 0)
		return;

	hash_seq_init(&status, queryTable);

	while((query_entry = (QueryTableEntry *) hash_seq_search(&status)) != NULL)
	{
		if (query_entry->status == QUERY_BLOCKED)
		{
			listing = lappend(listing, query_entry);
		}
		else if (query_entry->status == QUERY_GIVE_UP)
		{
			query_entry->status = QUERY_BLOCKED;
			giving_up++;
		}
	}

	if (listing == NIL)
		return;

	if(list_length(listing) > 1){
		elog(LOG, "Querying server for decision.");
		foreach(curr, listing){
			query_entry = (QueryTableEntry *) lfirst(curr);
			sendRequest(query_entry);
		}
		memset(endTensor, -1, QUERY_EMBEDDING_SIZE * sizeof(double));
		sendMsg(decision_server_socket, endTensor, sizeof(payload));
		elog(LOG, "%d queries sent for decision, %d queries is giving up.", list_length(listing), giving_up);
		decision = recvDecision();
		elog(LOG, "Received decision: %d", decision);
	}

	query_entry = (QueryTableEntry *)list_nth(listing, decision);
	query_entry->status = QUERY_AVAILABLE;
	query_entry->start_time = clock();

	state->runningQuery++;
}

/* TODO: Implement a heuristic based rescheduling algorithm*/
/* I noticed that some uneffective strategy will cause additionally deadlock.*/
void
Reschedule(HTAB *queryTable, ScheduleState *state)
{
	int avaliable;
	avaliable = MAX_CONCURRENT_QUERY - state->runningQuery;
	if (avaliable == 1){
		RescheduleWithServer(queryTable, state);
	}else{
		RescheduleUseHotTableFirst(queryTable, state);
	}
}

void
RescheduleUseHotTableFirst(HTAB *queryTable, ScheduleState *state)
{
	HASH_SEQ_STATUS status;
	QueryTableEntry *query_entry;
	List *listing = NIL;
	List *sorted = NIL;
	List *hot_score = NIL;
	ListCell *curr;
	HASHCTL hash_ctl;
	HTAB *table_ref;
	TableRef *table;
	int temp;
	int i, j, avaliable = 0;

	avaliable = MAX_CONCURRENT_QUERY - state->runningQuery;

	if (avaliable <= 0)
		return;

	hash_seq_init(&status, queryTable);

	memset(&hash_ctl, 0, sizeof(hash_ctl));
	hash_ctl.keysize = sizeof(Oid);
	hash_ctl.entrysize = sizeof(TableRef);

	table_ref = hash_create("TableRef", 10, &hash_ctl, HASH_ELEM | HASH_BLOBS);

	while ((query_entry = (QueryTableEntry *) hash_seq_search(&status)) != NULL)
	{
		listing = lappend(listing, query_entry);
		for (j = 0; j < MAX_AFFECTED_TABLE && query_entry->affected_tables[j] != 0; j++)
		{
			table = hash_search(table_ref, &query_entry->affected_tables[j], HASH_ENTER, NULL);
			table->num_of_ref++;
		}
	}

	foreach (curr, listing)
	{
		temp = 0;
		query_entry = (QueryTableEntry *) lfirst(curr);
		for (j = 0; j < MAX_AFFECTED_TABLE && query_entry->affected_tables[j] != 0; j++)
		{
			table = hash_search(table_ref, &query_entry->affected_tables[j], HASH_ENTER, NULL);
			temp += table->num_of_ref;
		}

		if (sorted == NIL)
		{
			sorted = lappend(sorted, query_entry);
			hot_score = lappend_int(hot_score, temp);
			continue;
		}

		for (i = 0; i < list_length(sorted); i++)
		{
			if (list_nth_int(hot_score, i) < temp)
			{
				break;
			}
		}

		sorted = list_insert_nth(sorted, i, query_entry);
		hot_score = list_insert_nth_int(hot_score, i, temp);
	}

	foreach (curr, sorted)
	{
		query_entry = (QueryTableEntry *) lfirst(curr);
		if (query_entry->status == QUERY_BLOCKED)
		{
			query_entry->status = QUERY_AVAILABLE;
			state->runningQuery++;
			avaliable = MAX_CONCURRENT_QUERY - state->runningQuery;
		}
		else if (query_entry->status == QUERY_GIVE_UP)
		{
			query_entry->status = QUERY_BLOCKED;
		}
		if (avaliable <= 0 || state->runningQuery == state->querynum)
			break;
	}

	hash_destroy(table_ref);
	list_free(listing);
	list_free(sorted);
	list_free(hot_score);
}

void
RescheduleUseFCFS(HTAB *queryTable, ScheduleState *state)
{
	HASH_SEQ_STATUS status;
	QueryTableEntry *query_entry;
	int avaliable;

	avaliable = MAX_CONCURRENT_QUERY - state->runningQuery;

	if (avaliable <= 0)
		return;

	hash_seq_init(&status, queryTable);
	while ((query_entry = (QueryTableEntry *) hash_seq_search(&status)) != NULL)
	{
		if (query_entry->status == QUERY_BLOCKED)
		{
			query_entry->status = QUERY_AVAILABLE;
			state->runningQuery++;
			avaliable = MAX_CONCURRENT_QUERY - state->runningQuery;
		}
		else if (query_entry->status == QUERY_GIVE_UP)
		{
			query_entry->status = QUERY_BLOCKED;
		}

		if (avaliable <= 0 || state->runningQuery == state->querynum)
		{
			hash_seq_term(&status);
			return;
		}
	}
}

void
RescheduleUseMinTableAffected(HTAB *queryTable, ScheduleState *state)
{
	HASH_SEQ_STATUS status;
	QueryTableEntry *query_entry;
	List *sorted = NIL;
	List *num_of_affected_tables = NIL;
	ListCell *curr;
	int j, index;
	int allowed = 0;

	if (allowed + state->runningQuery == MAX_CONCURRENT_QUERY)
		return;

	hash_seq_init(&status, queryTable);

	while ((query_entry = (QueryTableEntry *) hash_seq_search(&status)) != NULL)
	{
		for (j = 0; query_entry->affected_tables[j] != 0; j++)
			;
		index = 0;
		foreach (curr, sorted)
		{
			if (list_nth_int(num_of_affected_tables, index) > j)
				break;
			index++;
		}
		sorted = list_insert_nth(sorted, index, query_entry);
		num_of_affected_tables = list_insert_nth_int(num_of_affected_tables, index, j);
	}

	foreach (curr, sorted)
	{
		query_entry = (QueryTableEntry *) lfirst(curr);
		if (query_entry->status == QUERY_BLOCKED)
		{
			allowed++;
			query_entry->status = QUERY_AVAILABLE;
			state->runningQuery++;
		}
		else if (query_entry->status == QUERY_GIVE_UP)
		{
			query_entry->status = QUERY_BLOCKED;
		}
		if (allowed + state->runningQuery >= MAX_CONCURRENT_QUERY ||
			state->runningQuery == state->querynum)
			break;
	}
}

int embed_plan_node(double *tensor, Plan *plan, int start){

	int return_index = 0;

	if (start > QUERY_EMBEDDING_SIZE - 8){
		elog(ERROR, "Query embedding size is too small");
		return start;
	}

	//not using one-hot encoding here, becuase there are too many(~40) plan node types.
	tensor[start] = nodeTag(plan);

	tensor[start + 1] = plan->startup_cost;
	tensor[start + 2] = plan->total_cost;
	tensor[start + 3] = plan->plan_rows;
	tensor[start + 4] = plan->plan_width;
	tensor[start + 5] = plan->parallel_aware;
	tensor[start + 6] = plan->parallel_safe;
	tensor[start + 7] = plan->async_capable;

	return_index = start + 8;

	if (plan->lefttree)
		return_index = embed_plan_node(tensor, plan->lefttree, start + 8);
	if (plan->righttree)
		return_index = embed_plan_node(tensor, plan->righttree, start + 8 + 8);

	return return_index;
}


