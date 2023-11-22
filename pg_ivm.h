/*-------------------------------------------------------------------------
 *
 * pg_ivm.h
 *	  incremental view maintenance extension
 *
 * Portions Copyright (c) 1996-2022, PostgreSQL Global Development Group
 * Portions Copyright (c) 2022, IVM Development Group
 *
 *-------------------------------------------------------------------------
 */

#ifndef _PG_IVM_H_
#define _PG_IVM_H_

#include "catalog/objectaddress.h"
#include "fmgr.h"
#include "nodes/params.h"
#include "nodes/pathnodes.h"
#include "parser/parse_node.h"
#include "tcop/dest.h"
#include "utils/queryenvironment.h"
#include "storage/lwlock.h"
#include "access/transam.h"
#include "utils/elog.h"
#include "nodes/plannodes.h"
#include "utils/hsearch.h"
#include "executor/execdesc.h"

#define Natts_pg_ivm_immv 3

#define Anum_pg_ivm_immv_immvrelid 1
#define Anum_pg_ivm_immv_viewdef 2
#define Anum_pg_ivm_immv_ispopulated 3

#define IVM_LOG_LEVEL DEBUG1
/* pg_ivm.c */

extern void CreateChangePreventTrigger(Oid matviewOid);
extern Oid PgIvmImmvRelationId(void);
extern Oid PgIvmImmvPrimaryKeyIndexId(void);
extern bool isImmv(Oid immv_oid);

/* createas.c */

extern ObjectAddress ExecCreateImmv(ParseState *pstate, CreateTableAsStmt *stmt,
									ParamListInfo params, QueryEnvironment *queryEnv,
									QueryCompletion *qc);
extern void CreateIvmTriggersOnBaseTables(Query *qry, Oid matviewOid);
extern void CreateIndexOnIMMV(Query *query, Relation matviewRel);
extern Query *rewriteQueryForIMMV(Query *query, List *colNames);
extern void makeIvmAggColumn(ParseState *pstate, Aggref *aggref, char *resname,
							 AttrNumber *next_resno, List **aggs);

/* matview.c */

extern Query *get_immv_query(Relation matviewRel);
extern ObjectAddress ExecRefreshImmv(const RangeVar *relation, bool skipData,
									 const char *queryString, QueryCompletion *qc);
extern bool ImmvIncrementalMaintenanceIsEnabled(void);
extern Query *get_immv_query(Relation matviewRel);
extern Datum IVM_immediate_before(PG_FUNCTION_ARGS);
extern Datum IVM_immediate_maintenance(PG_FUNCTION_ARGS);
extern Query *rewrite_query_for_exists_subquery(Query *query);
extern Datum ivm_visible_in_prestate(PG_FUNCTION_ARGS);
extern void AtAbort_IVM(void);
extern char *getColumnNameStartWith(RangeTblEntry *rte, char *str, int *attnum);
extern bool isIvmName(const char *s);

/* ruleutils.c */

extern char *pg_ivm_get_viewdef(Relation immvrel, bool pretty);

/* subselect.c */
extern void inline_cte(PlannerInfo *root, CommonTableExpr *cte);

/* Learned_ivm-related Structures*/

/*
 * The current implementation heavily depends on shared memory,
 * however, the size of shared memory is fixed after allocation.
 * Therefore, we need to use fixed size for all the data structures
 * which is not scalable. Maybe we should seek for a better solution.
 */

#define QUERY_BLOCKED 0
#define QUERY_AVAILABLE 1
#define QUERY_ALLOWED 2
#define QUERY_GIVE_UP 3

/* Configurable parameters */
#define MAX_QUERY_NUM 1000
#define MAX_QUERY_LENGTH ((Size) 8192)
#define MAX_AFFECTED_TABLE 100

#define MAX_CONCURRENT_QUERY 4

#define HASH_TABLE_SIZE (MAX_QUERY_NUM * sizeof(QueryTableEntry))

typedef struct QueryTableKey
{
	char query_string[MAX_QUERY_LENGTH];
	uint32 pid;
} QueryTableKey;

/* Data Structure for metadata like quries, affected tables, immvs or something else */
/* Just a Proof of Concept for now, We should design a better structure saving them.*/
/* Considering to make this a hashmap */
/* Working in Progress */
typedef struct QueryTableEntry
{
	QueryTableKey key;
	Oid affected_tables[MAX_AFFECTED_TABLE];
	int status;
	TransactionId xid;
} QueryTableEntry;

/* Saving all necessary information we need for query scheduling*/
typedef struct SchedueState
{
	int querynum;
	int runningQuery;

	/* We only need one lock here, since the change to QueryTable
				will subsequently affect ScheduleTable
	*/
	LWLock *lock;
	int query_status[MAX_QUERY_NUM];

} ScheduleState;

#define SEGMENT_SIZE (sizeof(ScheduleState))

/* querysched.c */

extern QueryTableEntry *LogQuery(HTAB *queryTable, ScheduleState *state, PlannedStmt *plannedstmt,
								 const char *query_string);
extern void Reschedule(HTAB *queryTable, ScheduleState *state);
extern void RemoveLoggedQuery(QueryDesc *queryDesc, HTAB *queryHashTable,
							  ScheduleState *schedule_state);

#endif
