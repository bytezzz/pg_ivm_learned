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
#include "cJSON.h"
#include <netinet/in.h>
#include <resolv.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>



static void *context;
static void *requester;
extern int decision_server_socket;
extern int getIndexForTableEmbeeding(Oid oid);

void RescheduleUseMinTableAffected(HTAB *queryTable, ScheduleState *state, env_features *env_features);
void RescheduleRandomly(HTAB *queryTable, ScheduleState *state, env_features *env_features);
void RescheduleUseHotTableFirst(HTAB *queryTable, ScheduleState *state, env_features *env_features);
void RescheduleWithServer(HTAB *queryTable, ScheduleState *state, env_features *env_features);
int embed_plan_node(double *tensor, Plan *plan, int start);
response recvDecision(void);
static void connect_zmq();


static void connect_zmq()
{
	context = zmq_ctx_new();
	requester = zmq_socket(context, ZMQ_REQ);
	zmq_connect(requester, "tcp://localhost:5555");
}

/* Receive a decision from decision server. */
response
recvDecision(void)
{
	char buffer[256];
	cJSON *root;
	response response;

	zmq_recv(requester, &buffer, 256, 0);
	root = cJSON_Parse(buffer);
	response.decision = cJSON_GetObjectItem(root, "decision")->valueint;
	response.decision_id = cJSON_GetObjectItem(root, "decision_id")->valueint;

	elog(IVM_LOG_LEVEL, "Received decision: %d", response.decision);
	return response;
}

/* Log necessary information of a given query into the global data structure. */
void
LogQuery(ScheduleState *state, QueryDesc *desc, QueryTableEntry *query_entry)
{
	//int oidIndex;
	//ListCell *curr;
	//int effective_index = 0;
	//double *tensor;
	//int table_one_hot_index;
	//RangeTblEntry *modifyingRelation;
	PlannedStmt *plannedStmt;
	const char * query_string;
	char *json_plan;

	plannedStmt = desc->plannedstmt;
	query_string = desc->sourceText;

	/* Copy information.*/
	memset(query_entry, 0, sizeof(QueryTableEntry));
	strcpy(query_entry->key.query_string, query_string);
	query_entry->key.pid = MyProcPid;
	query_entry->status = QUERY_BLOCKED;
	query_entry->xid = GetCurrentTransactionId();

	state->querynum++;

	elog(IVM_LOG_LEVEL, "Logging Transactionid: %u", query_entry->xid);
	json_plan = plan_to_json(plannedStmt);
	strcpy(query_entry->query_plan_json, json_plan);

	free(json_plan);

	return;
}

/* Remove a query from the Global Hashtable.*/
void
UpdateStateForRemoving(ScheduleState *state, QueryDesc *desc, QueryTableEntry *query_entry, bool is_running)
{
	elog(IVM_LOG_LEVEL, "Removing Query xid:%d", query_entry->xid);

	state->querynum--;
	if (is_running){
		state->runningQuery--;
	}

	Assert(state->runningQuery >= 0 &&
		   state->runningQuery <= MAX_CONCURRENT_QUERY);
}

/*
 * This function will change the status of queries in the global hashtable
 * By setting some of their status into "QUERY_AVAILABLE"
 * corresponding backend processes will be able to execute them.
 *
 */
void
Reschedule(HTAB *queryTable, ScheduleState *state, ScheduleTag tag, env_features *env_features)
{
	int avaliable;
	avaliable = MAX_CONCURRENT_QUERY - state->runningQuery;

	env_features->schedule_tag = tag;

	/*
	 * The current implementation of RL algorithms can only make one decision at a time.
	 * This should not be a problem, as in concurrent scenarios,
	 * requests should always outnumber resources.
	 * Therefore, this function will be called whenever a new slot becomes available.
	 **/

	elog(LOG, "Rescheduling, %d queries are running, %d queries logged",
		 state->runningQuery,
		 state->querynum);

	if (true || avaliable == 1)
	{
		RescheduleWithServer(queryTable, state, env_features);
	}
	else
	{
		RescheduleUseMinTableAffected(queryTable, state, env_features);
	}
	Assert(state->runningQuery >= 0 && state->runningQuery <= MAX_CONCURRENT_QUERY);
}

/* Start query execution based on the decisions made by the remote server */
void
RescheduleWithServer(HTAB *queryTable, ScheduleState *state, env_features *env_features)
{
	HASH_SEQ_STATUS status;
	QueryTableEntry *query_entry;
	int available_num;
	response decision;
	List *listing = NIL;
	ListCell *curr = NULL;
	int giving_up = 0;
	cJSON *query;
	cJSON *plans;
	cJSON *parsed_plan;
	char *str;

	available_num = MAX_CONCURRENT_QUERY - state->runningQuery;

	if (available_num <= 0)
		return;

	if (requester == NULL)
	{
		connect_zmq();
	}

	hash_seq_init(&status, queryTable);

	/* give all queries an order, for fulture decision making */
	while ((query_entry = (QueryTableEntry *) hash_seq_search(&status)) != NULL)
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

	/* No avaliable query */
	if (listing == NIL)
		return;

  /* if currently have more than 1 queries to select from, query the server */
	if (list_length(listing) >= 1)
	{
		query = cJSON_CreateObject();
		plans= cJSON_CreateArray();
		elog(LOG, "Querying server for decision.");
		foreach (curr, listing)
		{
			query_entry = (QueryTableEntry *) lfirst(curr);
			parsed_plan = cJSON_Parse(query_entry->query_plan_json);
			cJSON_AddItemToArray(plans, parsed_plan);
		}

		cJSON_AddItemToObject(query, "plans", plans);
		cJSON_AddItemToObject(query, "env", env_to_json(env_features));

		str = cJSON_Print(query);
		zmq_send(requester, str, strlen(str), 0);

		cJSON_Delete(query);

		elog(LOG,
			 "%d queries sent for decision, %d queries is giving up.",
			 list_length(listing),
			 giving_up);

		decision = recvDecision();
		elog(LOG, "Received decision: %d", decision.decision);
	}else{
		decision.decision = 0;
		elog(LOG, "Only one query is available, start it directly.");
	}

	/* Start the picked query */
	query_entry = (QueryTableEntry *) list_nth(listing, decision.decision);
	query_entry->status = QUERY_AVAILABLE;
	query_entry->start_time = clock();
	query_entry->wakeup_by = decision.decision_id;

	list_free(listing);

	state->runningQuery++;
	elog(LOG, "Starting pid:%d, currently have %d runningQuery", query_entry->key.pid, state->runningQuery);
}

typedef struct TableRef
{
	Oid oid;
	int num_of_ref;
} TableRef;

/* This function will count all the blocked queries' access to tables
 * and then prioritize executing those queries that have accessed
 * the most frequently used tables with the highest popularity.*/
void
RescheduleUseHotTableFirst(HTAB *queryTable, ScheduleState *state, env_features *env)
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
			query_entry->wakeup_by = -1;
			elog(LOG, "Starting pid:%d, currently have %d runningQuery", query_entry->key.pid, state->runningQuery);
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
RescheduleRandomly(HTAB *queryTable, ScheduleState *state, env_features *env_features)
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
			query_entry->wakeup_by = -1;
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

/* This function will prioritize start queries that involve the fewest tables. */
void
RescheduleUseMinTableAffected(HTAB *queryTable, ScheduleState *state, env_features *env_features)
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
			elog(LOG, "Starting pid:%d, currently have %d runningQuery", query_entry->key.pid, state->runningQuery);
			query_entry->wakeup_by = -1;
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
