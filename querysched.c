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
#include "storage/procarray.h"
#include "conf.h"
#include "utils/lsyscache.h"

static void *context;
static void *requester;
extern HTAB *lockedRelations;
extern int getIndexForTableEmbeeding(Oid oid);

void RescheduleUseMinTableAffected(HTAB *queryTable, ScheduleState *state,
								   env_features *env_features, bool collect_exp);
void RescheduleRandomly(HTAB *queryTable, ScheduleState *state, env_features *env_features);
void RescheduleUseHotTableFirst(HTAB *queryTable, ScheduleState *state, env_features *env_features);
void RescheduleWithServer(HTAB *queryTable, ScheduleState *state, env_features *env_features);
int embed_plan_node(double *tensor, Plan *plan, int start);
response recvDecision(void);
bool isRunnable(QueryTableEntry *query_entry, TransactionId caller);
bool RelationAccessable(Oid oid, TransactionId caller);
static void connect_zmq();

static void
connect_zmq()
{
	context = zmq_ctx_new();
	requester = zmq_socket(context, ZMQ_REQ);
	zmq_connect(requester, "tcp://localhost:5555");
}

/* Receive a decision from decision server. */
response
recvDecision(void)
{
	char buffer[256] = {0};
	cJSON *root;
	response response;

	zmq_recv(requester, buffer, 256, 0);
	root = cJSON_Parse(buffer);
	response.decision = cJSON_GetObjectItem(root, "action")->valueint;
	response.decision_id = cJSON_GetObjectItem(root, "seq")->valueint;

	elog(IVM_LOG_LEVEL, "Received decision: %d", response.decision);
	return response;
}

/* Log necessary information of a given query into the global data structure. */
void
LogQuery(ScheduleState *state, QueryDesc *desc, QueryTableEntry *query_entry)
{
	int oidIndex;
	ListCell *curr;
	// int effective_index = 0;
	// double *tensor;
	// int table_one_hot_index;
	// RangeTblEntry *modifyingRelation;
	PlannedStmt *plannedStmt;
	const char *query_string;
	char *json_plan;

	plannedStmt = desc->plannedstmt;
	query_string = desc->sourceText;

	/* Copy information.*/
	memset(query_entry, 0, sizeof(QueryTableEntry));
	strcpy(query_entry->key.query_string, query_string);
	query_entry->key.pid = MyProcPid;
	query_entry->status = QUERY_BLOCKED;
	query_entry->xid = GetCurrentTransactionId();
	query_entry->num_of_relation_access = list_length(desc->plannedstmt->relationOids);

	state->querynum++;

	elog(IVM_LOG_LEVEL, "Logging Transactionid: %u, %s", query_entry->xid, desc->sourceText);
	json_plan = plan_to_json(plannedStmt);
	strcpy(query_entry->query_plan_json, json_plan);

	free(json_plan);

	oidIndex = 0;
	foreach (curr, plannedStmt->rtable)
	{
		query_entry->affected_tables[oidIndex++] = ((RangeTblEntry *) lfirst(curr))->relid;
	}

	return;
}

/* Remove a query from the Global Hashtable.*/
void
UpdateStateForRemoving(ScheduleState *state, QueryDesc *desc, QueryTableEntry *query_entry,
					   bool finished)
{
	cJSON *reward_json;
	char *str;
	char ack[8] = {0};
	double reward;

	elog(IVM_LOG_LEVEL, "Removing Query xid:%d", query_entry->xid);

	state->querynum--;
	if (finished)
	{
		state->runningQuery--;

		if (query_entry->wakeup_by != -1)
		{
			reward = (query_entry->outside_finished + 1) * 1.0 /
					 ((clock() - query_entry->start_time));
			reward_json = cJSON_CreateObject();
			cJSON_AddStringToObject(reward_json, "type", "reward");
			cJSON_AddNumberToObject(reward_json, "seq", query_entry->wakeup_by);
			cJSON_AddNumberToObject(reward_json,
									"reward",
									reward);
			str = cJSON_Print(reward_json);
			zmq_send(requester, str, strlen(str), 0);
			cJSON_Delete(reward_json);
			zmq_recv(requester, ack, 8, 0);

			if (strcmp(ack, "ack") != 0)
			{
				elog(ERROR, "Failed to receive ack from server.");
			}
		}
	}

	Assert(state->runningQuery >= 0 && state->runningQuery <= MAX_CONCURRENT_QUERY);
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
	if (requester == NULL)
	{
		connect_zmq();
	}

	avaliable = MAX_CONCURRENT_QUERY - state->runningQuery;

	env_features->schedule_tag = tag;

	/*
	 * The current implementation of RL algorithms can only make one decision at a time.
	 * This should not be a problem, as in concurrent scenarios,
	 * requests should always outnumber resources.
	 * Therefore, this function will be called whenever a new slot becomes available.
	 **/

	/*
	elog(LOG,
		 "Rescheduling, %d queries are running, %d queries logged",
		 state->runningQuery,
		 state->querynum);
	*/

	if (avaliable == 1)
	{
		RescheduleWithServer(queryTable, state, env_features);
	}
	else
	{
		RescheduleUseMinTableAffected(queryTable, state, env_features, false);
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
	//int index;
	response decision;
	List *blocking = NIL;
	List *running = NIL;
	List *idx = NIL;
	ListCell *curr = NULL;
	cJSON *query;
	cJSON *blocked_plans;
	cJSON *running_plans;
	cJSON *parsed_plan;
	char *str;

	available_num = MAX_CONCURRENT_QUERY - state->runningQuery;

	if (available_num <= 0)
		return;

	hash_seq_init(&status, queryTable);

	/* give all queries an order, for fulture decision making */
	while ((query_entry = (QueryTableEntry *) hash_seq_search(&status)) != NULL)
	{
		if (query_entry->status == QUERY_BLOCKED && isRunnable(query_entry, query_entry->xid))
		{
			blocking = lappend(blocking, query_entry);
		}
		else if (query_entry->status == QUERY_AVAILABLE)
		{
			running = lappend(running, query_entry);
		}
	}

	/* No avaliable query */
	if (blocking == NIL)
		return;

	/* if currently have more than 1 queries to select from, query the server */
	if (list_length(blocking) > 1)
	{
		query = cJSON_CreateObject();
		blocked_plans = cJSON_CreateArray();
		running_plans = cJSON_CreateArray();
		elog(LOG, "Querying server for decision.");

		foreach (curr, blocking)
		{
			query_entry = (QueryTableEntry *) lfirst(curr);
			parsed_plan = cJSON_Parse(query_entry->query_plan_json);
			cJSON_AddItemToArray(blocked_plans, parsed_plan);
		}

		foreach (curr, running)
		{
			query_entry = (QueryTableEntry *) lfirst(curr);
			parsed_plan = cJSON_Parse(query_entry->query_plan_json);
			cJSON_AddItemToArray(running_plans, parsed_plan);
		}

		cJSON_AddItemToObject(query, "candidate_plans", blocked_plans);
		cJSON_AddItemToObject(query, "running_plans", running_plans);
		cJSON_AddItemToObject(query, "env", env_to_json(env_features));
		cJSON_AddStringToObject(query, "type", "schedule");

		str = cJSON_Print(query);
		zmq_send(requester, str, strlen(str), 0);

		cJSON_Delete(query);

		elog(LOG,
			 "%d queries sent for decision,  %d queries are running.",
			 list_length(blocking),
			 list_length(running));

		decision = recvDecision();
		elog(LOG, "Received decision: %d", decision.decision);
	}
	else
	{
		decision.decision = 0;
		decision.decision_id = -1;
		elog(LOG, "Only one query is available, start it directly.");
	}

	/* Start the picked query */
	query_entry = (QueryTableEntry *) list_nth(blocking, decision.decision);
	query_entry->status = QUERY_AVAILABLE;
	query_entry->wakeup_by = decision.decision_id;

	list_free(running);
	list_free(blocking);
	list_free(idx);

	state->runningQuery++;
	elog(LOG,
		 "Starting pid:%d, currently have %d runningQuery",
		 query_entry->key.pid,
		 state->runningQuery);
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
			elog(LOG,
				 "Starting pid:%d, currently have %d runningQuery",
				 query_entry->key.pid,
				 state->runningQuery);
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
RescheduleUseMinTableAffected(HTAB *queryTable, ScheduleState *state, env_features *env_features,
							  bool collect_exp)
{
	HASH_SEQ_STATUS status;
	QueryTableEntry *query_entry;
	List *sorted = NIL;
	List *num_of_affected_tables = NIL;
	List *running = NIL;
	List *blocked = NIL;
	void *to_delete;
	cJSON *query_json;
	cJSON *parsed;
	cJSON *blocked_plans;
	cJSON *running_plans;
	char *str;
	char fake_decision_received[256];
	ListCell *curr, *tmp;
	int j, index;
	int allowed = 0;

	if (allowed + state->runningQuery == MAX_CONCURRENT_QUERY)
		return;

	hash_seq_init(&status, queryTable);

	while ((query_entry = (QueryTableEntry *) hash_seq_search(&status)) != NULL)
	{
		j = query_entry->num_of_relation_access;
		index = 0;
		foreach (curr, sorted)
		{
			if (list_nth_int(num_of_affected_tables, index) > j)
				break;
			index++;
		}
		sorted = list_insert_nth(sorted, index, query_entry);
		num_of_affected_tables = list_insert_nth_int(num_of_affected_tables, index, j);

		if (collect_exp){
			if (query_entry->status == QUERY_AVAILABLE)
			{
				running = lappend(running, query_entry);
			}
			else if (query_entry->status == QUERY_BLOCKED)
			{
				blocked = lappend(blocked, query_entry);
			}
		}

	}

	foreach (curr, sorted)
	{
		query_entry = (QueryTableEntry *) lfirst(curr);
		if (query_entry->status == QUERY_BLOCKED && isRunnable(query_entry, query_entry->xid))
		{
			to_delete = NULL;
			if (collect_exp){
				query_json = cJSON_CreateObject();
				blocked_plans = cJSON_CreateArray();
				running_plans = cJSON_CreateArray();

				foreach (tmp, running)
					{
						parsed = cJSON_Parse(((QueryTableEntry *) lfirst(tmp))->query_plan_json);
						cJSON_AddItemToArray(running_plans, parsed);
					}

				/*Bug: 此处Blocked List是在对序列排序的时候根据isRunnable收集的。我们再一次遍历这个list的时候，isRunnable结果已经变了
					因此也许目前准备启动的任务并不在Blocked List中。
					Proposal：Build Blocked List的时候不去检查isRunnable，而是只在这里检查。
				*/
				foreach (tmp, blocked)
					{
						if (!isRunnable(lfirst(tmp), query_entry->xid))
							continue;

						parsed = cJSON_Parse(((QueryTableEntry *) lfirst(tmp))->query_plan_json);
						cJSON_AddItemToArray(blocked_plans, parsed);
						if (query_entry == lfirst(tmp)){
							cJSON_AddNumberToObject(query_json, "action", cJSON_GetArraySize(blocked_plans) - 1);
							running = lappend(running, query_entry);
							to_delete = query_entry;
						}
					}
				if (to_delete)
					blocked = list_delete_ptr(blocked, to_delete);
				cJSON_AddItemToObject(query_json, "candidate_plans", blocked_plans);
				cJSON_AddItemToObject(query_json, "running_plans", running_plans);
				cJSON_AddItemToObject(query_json, "env", env_to_json(env_features));
				cJSON_AddStringToObject(query_json, "type", "schedule");
				str = cJSON_Print(query_json);
				zmq_send(requester, str, strlen(str), 0);
				//elog(LOG,"Query Sent");
				free(str);
				cJSON_Delete(query_json);
				zmq_recv(requester, fake_decision_received, 256, 0);
				query_entry->wakeup_by = cJSON_GetNumberValue(cJSON_GetObjectItem(cJSON_Parse(fake_decision_received), "seq"));
				//elog(LOG,"Received SEQ: %d", query_entry->wakeup_by);
			}
			allowed++;
			query_entry->status = QUERY_AVAILABLE;
			state->runningQuery++;
			elog(LOG,
				 "Starting pid:%d, currently have %d runningQuery",
				 query_entry->key.pid,
				 state->runningQuery);
			if (!collect_exp)
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

bool isRunnable(QueryTableEntry *query_entry, TransactionId caller_xid){
	Oid curr;
	RefedImmv *refed_immv;
	int i,j;

	for (i = 0; i < MAX_AFFECTED_TABLE; i++){
		curr = query_entry->affected_tables[i];
		if (curr == 0){
			break;
		}
		refed_immv = getRefrenceImmv(curr);
		if (refed_immv == NULL)
			continue;
		for(j = 0 ; j < refed_immv->refed_table_num; j++){
			if (!RelationAccessable(refed_immv->refed_table[j], caller_xid)){
				return false;
			}
		}
	}
	return true;
}

bool RelationAccessable(Oid oid, TransactionId caller_xid){
	LockedTableKey key;
	LockedTableEntry *entry;
	bool found;
	memset(&key, 0, sizeof(LockedTableKey));
	key.table = oid;
	entry = (LockedTableEntry*) hash_search(lockedRelations, &key, HASH_FIND, &found);

	if (!found || entry->held_by == caller_xid){
		return true;
	}

	if (TransactionIdIsInProgress(entry->held_by)){
		return false;
	}

	return true;
}

void AddLockInfo(Oid oid, LOCKMODE lockmode){
	LockedTableKey key;
	LockedTableEntry *entry;
	bool found;
	char *relname;

	if(lockmode == 0){
		return;
	}
	memset(&key, 0, sizeof(LockedTableKey));
	key.table = oid;
	entry = (LockedTableEntry*) hash_search(lockedRelations, &key, HASH_ENTER, &found);

	if (found && TransactionIdIsInProgress(entry->held_by) && entry->held_by != GetTopTransactionId() && entry->mode >= lockmode){
		elog(ERROR, "Table %d is already locked", oid);
	}

	entry->held_by = GetTopTransactionId();
	entry->mode = lockmode;
	relname = get_rel_name(oid);
	elog(LOG, "Table %s is now locked by %d in %d mode", relname, entry->held_by, lockmode);
}

void RemoveLockInfo(Oid oid){
	LockedTableKey key;
	bool found;
	memset(&key, 0, sizeof(LockedTableKey));
	key.table = oid;
	hash_search(lockedRelations, &key, HASH_REMOVE, &found);
	if (!found){
		elog(ERROR, "Table %d is not locked", oid);
	}
	elog(LOG, "Table %d is unlocked", oid);
}
