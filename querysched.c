#include "postgres.h"
#include "storage/lwlock.h"
#include "utils/elog.h"
#include "stdlib.h"
#include "miscadmin.h"
#include "parser/parsetree.h"
#include "access/xact.h"

#include "pg_ivm.h"

static void LogAffectedTablesRecurse(Query *, Node *, ScheduleState *, Relids *, int);
static void LogAffectedTables(Query *qry, ScheduleState *state, int index);
static void LogTableOid(ScheduleState *state, Oid oid, int index);

int
LogQuery(ScheduleState *state, Query *query, const char *query_string)
{
	int index;
	QueryTableEntry *query_entry;

	for (index = 0; index < MAX_QUERY_NUM; index++)
	{
		if (state->queryTable.queries[index].xid == 0)
			break;
	}

	if (index == MAX_QUERY_NUM - 1)
		ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("Too many queries in the system")));

	state->querynum++;

	query_entry = &state->queryTable.queries[index];

	strcpy(query_entry->query_string, query_string);
	query_entry->xid = GetCurrentTransactionId();

	elog(IVM_LOG_LEVEL,
		 "Logging id: %d, Transactionid: %u, Query: %s",
		 index,
		 query_entry->xid,
		 query_string);
	LogAffectedTables(query, state, index);

	Reschedule(state);
	return index;
}

/* This function immitate PG_IVM's CreateIvmTriggersOnBaseTables*/
static void
LogAffectedTables(Query *qry, ScheduleState *state, int index)
{
	Relids relids = NULL;
	int i;

	/* Immediately return if we don't have any base tables. */
	if (list_length(qry->rtable) < 1)
		return;

	LogAffectedTablesRecurse(qry, (Node *) qry, state, &relids, index);

	for (i = 0; i < MAX_AFFECTED_TABLE; i++)
	{
		if (state->queryTable.queries[index].affected_tables[i] == 0)
			break;

		elog(IVM_LOG_LEVEL,
			 "Affected table: %d",
			 state->queryTable.queries[index].affected_tables[i]);
	}

	bms_free(relids);
}

/* This function immitate PG_IVM's CreateIvmTriggersOnBaseTablesRecurse*/
static void
LogAffectedTablesRecurse(Query *qry, Node *node, ScheduleState *state, Relids *relids, int index)
{
	if (node == NULL)
		return;

	/* This can recurse, so check for excessive recursion */
	check_stack_depth();

	switch (nodeTag(node))
	{
		case T_Query:
		{
			Query *query = (Query *) node;
			ListCell *lc;

			LogAffectedTablesRecurse(qry, (Node *) query->jointree, state, relids, index);
			foreach (lc, query->cteList)
			{
				CommonTableExpr *cte = (CommonTableExpr *) lfirst(lc);
				Assert(IsA(cte->ctequery, Query));
				LogAffectedTablesRecurse((Query *) cte->ctequery,
										 cte->ctequery,
										 state,
										 relids,
										 index);
			}
		}
		break;

		case T_RangeTblRef:
		{
			int rti = ((RangeTblRef *) node)->rtindex;
			RangeTblEntry *rte = rt_fetch(rti, qry->rtable);

			if (rte->rtekind == RTE_RELATION && !bms_is_member(rte->relid, *relids))
			{
				LogTableOid(state, rte->relid, index);
				*relids = bms_add_member(*relids, rte->relid);
			}
			else if (rte->rtekind == RTE_SUBQUERY)
			{
				Query *subquery = rte->subquery;
				Assert(rte->subquery != NULL);
				LogAffectedTablesRecurse(subquery, (Node *) subquery, state, relids, index);
			}
		}
		break;

		case T_FromExpr:
		{
			FromExpr *f = (FromExpr *) node;
			ListCell *l;

			foreach (l, f->fromlist)
				LogAffectedTablesRecurse(qry, lfirst(l), state, relids, index);
		}
		break;

		case T_JoinExpr:
		{
			JoinExpr *j = (JoinExpr *) node;

			LogAffectedTablesRecurse(qry, j->larg, state, relids, index);
			LogAffectedTablesRecurse(qry, j->rarg, state, relids, index);
		}
		break;

		default:
			elog(ERROR, "unrecognized node type: %d", (int) nodeTag(node));
	}
}

static void
LogTableOid(ScheduleState *state, Oid oid, int index)
{
	int i;
	QueryTableEntry *query;

	query = &state->queryTable.queries[index];

	for (i = 0; i < MAX_AFFECTED_TABLE; i++)
	{
		if (query->affected_tables[i] == 0)
		{
			query->affected_tables[i] = oid;
			return;
		}
	}

	elog(ERROR,
		 "Too many affected tables for query: %s",
		 state->queryTable.queries[index].query_string);
}

/* TODO: Implement a heuristic based rescheduling algorithm*/
void
Reschedule(ScheduleState *state)
{
	int i;
	ScheduleTable *scheduleTable;

	scheduleTable = &state->scheduleTable;

	for (i = 0; i < MAX_QUERY_NUM; i++)
	{
		if (state->queryTable.queries[i].xid == 0)
			continue;

		scheduleTable->query_status[i] = QUERY_AVAILABLE;

		elog(IVM_LOG_LEVEL, "Allowing query: %d", i);
	}
}
