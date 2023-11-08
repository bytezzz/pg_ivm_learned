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

int
LogQuery(ScheduleState *state, PlannedStmt *query, const char *query_string)
{
	int index, oidIndex;
	QueryTableEntry *query_entry;
	ListCell *roid;

	for (index = 0; index < MAX_QUERY_NUM; index++)
	{
		if (state->queryTable[index].xid == 0)
			break;
	}

	if (index == MAX_QUERY_NUM - 1)
		ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("Too many queries in the system")));

	state->querynum++;

	query_entry = &state->queryTable[index];

	strcpy(query_entry->query_string, query_string);
	query_entry->xid = GetCurrentTransactionId();
	query_entry->queryId = query->queryId;
	query_entry->procId = MyProcPid;

	elog(IVM_LOG_LEVEL,
		 "Logging id: %d, Transactionid: %u, Query: %s, QueryId: %ld, myPid is: %d",
		 index,
		 query_entry->xid,
		 query_string,
		 query_entry->queryId,
		 MyProcPid);

	/* Not sure if this is correct.*/
	oidIndex = 0;
	foreach(roid, query->relationOids)
	{
		if (oidIndex > MAX_AFFECTED_TABLE){
			elog(ERROR,
				"Too many affected tables for query: %s",
				state->queryTable[index].query_string);
		}
		query_entry->affected_tables[oidIndex++] = lfirst_oid(roid);
		elog(IVM_LOG_LEVEL, "Logging affected table: %d", lfirst_oid(roid));
	}

	return index;
}

/* TODO: Implement a heuristic based rescheduling algorithm*/
void
Reschedule(ScheduleState *state){

	int i;

	for (i = 0; i < MAX_QUERY_NUM; i++)
	{
		if (state->queryTable[i].xid == 0)
			continue;

		state->query_status[i] = QUERY_AVAILABLE;
	}

}


