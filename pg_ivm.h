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

/* Learned_ivm-related Structures */

/* Decision server info */
#define SERVERNAME  "localhost"
#define PORT  2300

/*ivm related log level*/
#define IVM_LOG_LEVEL DEBUG1

enum query_staus{

	/* After a new received query logging into the global structure, it will become blocked*/
	QUERY_BLOCKED,

	/* Reschedule functions will change their status into "avaliable",
	 * after which corresponding backend process can execute them. */
	QUERY_AVAILABLE,

	/* If a query can not acquire all necessary locks, it will give up this schedule,
		queries in the status of giving up can not be start in next schedule decision,
		but will become avaliable after a schedule event.*/
	QUERY_GIVE_UP
};

/* Configurable parameters */
#define MAX_QUERY_NUM 1000
#define MAX_QUERY_LENGTH ((Size) 8192)
#define MAX_AFFECTED_TABLE 100

#define MAX_CONCURRENT_QUERY 4

#define QUERY_EMBEDDING_SIZE 1024

//8 base tables, 8 immvs, 1 for all other tables
#define WORKING_TABLES 17

#define HASH_TABLE_SIZE (MAX_QUERY_NUM * sizeof(QueryTableEntry))

/*
 * The current implementation heavily depends on shared memory,
 * however, the size of shared memory is fixed after allocation.
 * Therefore, we need to use fixed size for all the data structures
 * which is not scalable.
 * Maybe switch to the new dynamic shared memory features in fulture.
 */

typedef struct QueryTableKey
{
	char query_string[MAX_QUERY_LENGTH];
	uint32 pid;
} QueryTableKey;

typedef struct QueryTableEntry
{
	QueryTableKey key;
	Oid affected_tables[MAX_AFFECTED_TABLE];
	double embedding[QUERY_EMBEDDING_SIZE];
	int status;
	TransactionId xid;
	clock_t start_time;
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

	int tableAccessCounter[WORKING_TABLES];
} ScheduleState;

#define SEGMENT_SIZE (sizeof(ScheduleState))

/* querysched.c */

extern QueryTableEntry *LogQuery(HTAB *queryTable, ScheduleState *state, PlannedStmt *plannedstmt,
								 const char *query_string);
extern void Reschedule(HTAB *queryTable, ScheduleState *state);
extern void RemoveLoggedQuery(QueryDesc *queryDesc, HTAB *queryHashTable,
							  ScheduleState *schedule_state);

/* env_embedding.c */

extern void log_table_access(ScheduleState * ss, Oid *affected_table);

#endif
