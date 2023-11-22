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

#include "pg_ivm.h"

void RescheduleUseMinTableAffected(HTAB *queryTable, ScheduleState *state);
void RescheduleUseFCFS(HTAB *queryTable, ScheduleState *state);
void RescheduleUseHotTableFirst(HTAB *queryTable, ScheduleState *state);

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

	/* Not sure if this is correct.*/
	oidIndex = 0;
	foreach (roid, plannedStmt->relationOids)
	{
		if (oidIndex > MAX_AFFECTED_TABLE)
		{
			elog(ERROR, "Too many affected tables for query: %s", query_string);
		}
		query_entry->affected_tables[oidIndex++] = lfirst_oid(roid);
		// elog(IVM_LOG_LEVEL, "Logging affected table: %d", lfirst_oid(roid));
	}
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

/* TODO: Implement a heuristic based rescheduling algorithm*/
/* I noticed that some uneffective strategy will cause additionally deadlock.*/
void
Reschedule(HTAB *queryTable, ScheduleState *state)
{
	// RescheduleUseFCFS(queryTable, state);
	// RescheduleUseMinTableAffected(queryTable, state);
	RescheduleUseHotTableFirst(queryTable, state);
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
