#include "postgres.h"
#include "storage/lwlock.h"
#include "utils/elog.h"
#include "stdlib.h"
#include "miscadmin.h"
#include "parser/parsetree.h"
#include "access/xact.h"
#include "nodes/plannodes.h"
#include "utils/builtins.h"

#include "pg_ivm.h"

typedef struct ReferedTable
{
	Oid oid;
	int count;
} ReferedTable;

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

	elog(IVM_LOG_LEVEL,
		 "PID %d: Logging Transactionid: %u, Query: %s",
		 MyProcPid,
		 query_entry->xid,
		 query_string);

	/* Not sure if this is correct.*/
	oidIndex = 0;
	foreach (roid, plannedStmt->relationOids)
	{
		if (oidIndex > MAX_AFFECTED_TABLE)
		{
			elog(ERROR, "Too many affected tables for query: %s", query_string);
		}
		query_entry->affected_tables[oidIndex++] = lfirst_oid(roid);
		elog(IVM_LOG_LEVEL, "Logging affected table: %d", lfirst_oid(roid));
	}
	return query_entry;
}

/* TODO: Implement a heuristic based rescheduling algorithm*/
void
Reschedule(HTAB *queryTable, ScheduleState *state)
{
	HASH_SEQ_STATUS status;
	QueryTableEntry *query_entry;

	hash_seq_init(&status, queryTable);

	while ((query_entry = (QueryTableEntry *) hash_seq_search(&status)) != NULL)
	{
		if (query_entry->status == QUERY_BLOCKED)
		{
			elog(IVM_LOG_LEVEL,
				 "Allowing PID:%d's Query: %s",
				 query_entry->key.pid,
				 query_entry->key.query_string);
			query_entry->status = QUERY_AVAILABLE;
		}
	}
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

	if (strcmp(queryDesc->sourceText, "") == 0)
		goto exit;

	LWLockAcquire(schedule_state->lock, LW_EXCLUSIVE);

	query = hash_search(queryHashTable, &key, HASH_REMOVE, &found);

	if (!found || query == NULL)
	{
		elog(IVM_LOG_LEVEL, "Pid:%d: Cannot find Query:%s", MyProcPid, queryDesc->sourceText);
		goto exitWithReleasing;
	}
	schedule_state->querynum--;

exitWithReleasing:
	LWLockRelease(schedule_state->lock);

exit:
	return;
}
