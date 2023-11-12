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
LogQuery(HTAB *queryTable, ScheduleState *state, PlannedStmt *query, const char *query_string)
{
	int oidIndex;
	bool found;
	QueryTableEntry *query_entry;
	ListCell *roid;

	if (state->querynum >= MAX_QUERY_NUM)
		ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("Too many queries in the system")));

	state->querynum++;

	query_entry = (QueryTableEntry *) hash_search(queryTable, query_string, HASH_ENTER, &found);

	if (found)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DUPLICATE_OBJECT),
				 errmsg("Duplicate query id, current queryID: %ld, query_string: %s, before "
						"query_string: %s",
						query->queryId,
						query_string,
						query_entry->query_string)));
	}

	strcpy(query_entry->query_string, query_string);
	query_entry->status = QUERY_BLOCKED;
	query_entry->xid = GetCurrentTransactionId();
	query_entry->queryId = query->queryId;
	query_entry->procId = MyProcPid;

	elog(IVM_LOG_LEVEL,
		 "Logging Transactionid: %u, Query: %s, QueryId: %ld, myPid is: %d",
		 query_entry->xid,
		 query_string,
		 query_entry->queryId,
		 MyProcPid);

	/* Not sure if this is correct.*/
	oidIndex = 0;
	foreach (roid, query->relationOids)
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
				 "Rescheduling query: %s, hashkey: %ld",
				 query_entry->query_string,
				 query_entry->queryId);
			query_entry->status = QUERY_AVAILABLE;
		}
	}
}

void
RemoveLoggedQuery(QueryDesc *queryDesc, HTAB *queryHashTable, ScheduleState *schedule_state)
{
	QueryTableEntry *query;
	bool found;

	if (strcmp(queryDesc->sourceText, "") == 0)
		goto exit;

	LWLockAcquire(schedule_state->lock, LW_EXCLUSIVE);

	query = hash_search(queryHashTable, queryDesc->sourceText, HASH_FIND, &found);

	if (!found)
	{
		elog(IVM_LOG_LEVEL,
			 "Cannot find Query:%s. Qid %ld, myPid is: %d",
			 queryDesc->sourceText,
			 queryDesc->plannedstmt->queryId,
			 MyProcPid);
		goto exitWithReleasing;
	}

	elog(IVM_LOG_LEVEL,
		 "Query:%s Finished!. Removing Qid %ld, myPid is: %d",
		 queryDesc->sourceText,
		 query->queryId,
		 MyProcPid);

	if (MyProcPid == query->procId)
	{
		query = hash_search(queryHashTable, queryDesc->sourceText, HASH_REMOVE, &found);
		if (query == NULL)
			elog(ERROR, "Can not remove hash table entry");
		memset(query, 0, sizeof(QueryTableEntry));
		schedule_state->querynum--;
	}

exitWithReleasing:
	LWLockRelease(schedule_state->lock);

exit:
	return;
}
