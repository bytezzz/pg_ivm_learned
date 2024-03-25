/*-------------------------------------------------------------------------
 *
 * pg_ivm.c
 *	  incremental view maintenance extension
 *    Routines for user interfaces and callback functions
 *
 * Portions Copyright (c) 1996-2022, PostgreSQL Global Development Group
 * Portions Copyright (c) 2022, IVM Development Group
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "access/genam.h"
#include "access/table.h"
#include "access/xact.h"
#include "catalog/dependency.h"
#include "catalog/indexing.h"
#include "catalog/namespace.h"
#include "catalog/objectaccess.h"
#include "catalog/pg_namespace_d.h"
#include "catalog/pg_trigger_d.h"
#include "commands/trigger.h"
#include "parser/analyze.h"
#include "parser/parser.h"
#include "parser/scansup.h"
#include "tcop/tcopprot.h"
#include "tcop/utility.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/lsyscache.h"
#include "utils/regproc.h"
#include "utils/rel.h"
#include "utils/varlena.h"
#include "miscadmin.h"
#include "storage/ipc.h"
#include "optimizer/planner.h"
#include "executor/execdesc.h"
#include "executor/executor.h"
#include "access/parallel.h"
#include "storage/lmgr.h"
#include "catalog/catalog.h"

#include "pg_ivm.h"
#include "conf.h"

#include <netinet/in.h>
#include <resolv.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>

/*
	Macro to check if query order enforcement is needed,
	Only enforce the top level , non-utility query and not in parallel worker.
*/
#define enable_enforce(level) (!IsParallelWorker() && (level) == 0 && !isUtility)

PG_MODULE_MAGIC;

static Oid pg_ivm_immv_id = InvalidOid;
static Oid pg_ivm_immv_pkey_id = InvalidOid;

static object_access_hook_type PrevObjectAccessHook = NULL;
static shmem_request_hook_type PrevShmemRequestHook = NULL;
static shmem_startup_hook_type PrevShmemStartupHook = NULL;
static planner_hook_type PrevPlanHook = NULL;
static ExecutorStart_hook_type PrevExecutionStartHook = NULL;
static ExecutorFinish_hook_type PrevExecutionFinishHook = NULL;
static ExecutorEnd_hook_type PrevExecutionEndHook = NULL;
static ExecutorRun_hook_type prevExecutorRunHook = NULL;
static ProcessUtility_hook_type prev_ProcessUtility = NULL;

static ScheduleState *schedule_state = NULL;
static HTAB *queryHashTable = NULL;
HTAB *lockedRelations = NULL;


/* This help to determine if current running query is a top-level query or not
 * because these hooks are called recursively when a top-level query have sub queries. */
static int nesting_level = 0;

/*Socket of Python decision server*/
void *requester = NULL;
void *context = NULL;

/*
 * This is used to solve the problem of for-loop
 * expression in for-loop condition will be executed many times without calling ExecutorStart.
 * We use this variable to record the number of times ExecutorStart is called.
 * If it is not 0, we will not skip order enforcement in the following ExecutorRun Hook.
 * See regression test case insert_conflict for more details.
 */
static int full_process = 0;

/*
 * Flag to indicate if the current query is a utility command.
 * If it is, we will not do order enforcement.
 */
static bool isUtility = false;

void _PG_init(void);

static void IvmXactCallback(XactEvent event, void *arg);
static void IvmSubXactCallback(SubXactEvent event, SubTransactionId mySubid,
							   SubTransactionId parentSubid, void *arg);
static void parseNameAndColumns(const char *string, List **names, List **colNames);

static void PgIvmObjectAccessHook(ObjectAccessType access, Oid classId, Oid objectId, int subId,
								  void *arg);

static void pg_hook_shmem_request(void);
static void pg_hook_shmem_startup(void);
static PlannedStmt *pg_hook_planner(Query *parse, const char *query_string, int cursor_options,
									ParamListInfo bound_params);
void pg_hook_execution_start(QueryDesc *queryDesc, int eflags);
void pg_hook_executor_run(QueryDesc *queryDesc, ScanDirection direction, uint64 count,
						  bool execute_once);
void pg_hook_execution_finish(QueryDesc *queryDesc);
void pg_hook_executor_end(QueryDesc *queryDesc);
void pg_hook_process_utility(PlannedStmt *pstmt, const char *queryString, bool readOnlyTree,
							 ProcessUtilityContext context, ParamListInfo params,
							 QueryEnvironment *queryEnv, DestReceiver *dest, QueryCompletion *qc);


/* SQL callable functions */
PG_FUNCTION_INFO_V1(create_immv);
PG_FUNCTION_INFO_V1(refresh_immv);
PG_FUNCTION_INFO_V1(IVM_prevent_immv_change);
PG_FUNCTION_INFO_V1(get_immv_def);

/* learned ivm related funcs */
extern void sendMsg(int sock, void *msg, uint32_t msgsize);
void getLocksHeldByMe(StringInfo info);
void finishRunningQuery(QueryDesc *queryDesc);
void finishUnstartedQuery(QueryDesc *queryDesc);

static inline void
SetLocktagRelationOid(LOCKTAG *tag, Oid relid)
{
	Oid dbid;

	if (IsSharedRelation(relid))
		dbid = InvalidOid;
	else
		dbid = MyDatabaseId;

	SET_LOCKTAG_RELATION(*tag, dbid, relid);
}

/*
 * Call back functions for cleaning up
 */
static void
IvmXactCallback(XactEvent event, void *arg)
{
	if (event == XACT_EVENT_ABORT)
		AtAbort_IVM();
}

static void
IvmSubXactCallback(SubXactEvent event, SubTransactionId mySubid, SubTransactionId parentSubid,
				   void *arg)
{
	if (event == SUBXACT_EVENT_ABORT_SUB)
		AtAbort_IVM();
}

/*
 * Module load callback
 */
void
_PG_init(void)
{
	elog(LOG, "Initializing PG_LEARNED_IVM");
	RegisterXactCallback(IvmXactCallback, NULL);
	RegisterSubXactCallback(IvmSubXactCallback, NULL);

	elog(LOG, "Connecting to decision server");

	//decision_server_socket = connect_to_server(SERVERNAME, PORT);

	elog(LOG, "Connected to %s", SERVERNAME);

	PrevObjectAccessHook = object_access_hook;
	object_access_hook = PgIvmObjectAccessHook;

	/* Install hooks on shared memory allocation
	 * for necessary global data structures
	 */
	PrevShmemRequestHook = shmem_request_hook;
	shmem_request_hook = pg_hook_shmem_request;
	PrevShmemStartupHook = shmem_startup_hook;
	shmem_startup_hook = pg_hook_shmem_startup;

	/* Install hooks on planner and executor */
	PrevPlanHook = planner_hook;
	planner_hook = pg_hook_planner;

	PrevExecutionStartHook = ExecutorStart_hook;
	ExecutorStart_hook = pg_hook_execution_start;

	prevExecutorRunHook = ExecutorRun_hook;
	ExecutorRun_hook = pg_hook_executor_run;

	PrevExecutionFinishHook = ExecutorFinish_hook;
	ExecutorFinish_hook = pg_hook_execution_finish;

	PrevExecutionEndHook = ExecutorEnd_hook;
	ExecutorEnd_hook = pg_hook_executor_end;

	prev_ProcessUtility = ProcessUtility_hook;
	ProcessUtility_hook = pg_hook_process_utility;
}

/*
 * Given a C string, parse it into a qualified relation name
 * followed by a optional parenthesized list of column names.
 */
static void
parseNameAndColumns(const char *string, List **names, List **colNames)
{
	char *rawname;
	char *ptr;
	char *ptr2;
	bool in_quote;
	bool has_colnames = false;
	List *cols;
	ListCell *lc;

	/* We need a modifiable copy of the input string. */
	rawname = pstrdup(string);

	/* Scan to find the expected left paren; mustn't be quoted */
	in_quote = false;
	for (ptr = rawname; *ptr; ptr++)
	{
		if (*ptr == '"')
			in_quote = !in_quote;
		else if (*ptr == '(' && !in_quote)
		{
			has_colnames = true;
			break;
		}
	}

	/* Separate the name and parse it into a list */
	*ptr++ = '\0';
#if defined(PG_VERSION_NUM) && (PG_VERSION_NUM >= 160000)
	*names = stringToQualifiedNameList(rawname, NULL);
#else
	*names = stringToQualifiedNameList(rawname);
#endif

	if (!has_colnames)
		goto end;

	/* Check for the trailing right parenthesis and remove it */
	ptr2 = ptr + strlen(ptr);
	while (--ptr2 > ptr)
	{
		if (!scanner_isspace(*ptr2))
			break;
	}
	if (*ptr2 != ')')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("expected a right parenthesis")));

	*ptr2 = '\0';

	if (!SplitIdentifierString(ptr, ',', &cols))
		ereport(ERROR, (errcode(ERRCODE_INVALID_NAME), errmsg("invalid name syntax")));

	if (list_length(cols) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_OBJECT_DEFINITION),
				 errmsg("must specify at least one column name")));

	foreach (lc, cols)
	{
		char *colname = lfirst(lc);
		*colNames = lappend(*colNames, makeString(pstrdup(colname)));
	}

end:
	pfree(rawname);
}

/*
 * User interface for creating an IMMV
 */
Datum
create_immv(PG_FUNCTION_ARGS)
{
	text *t_relname = PG_GETARG_TEXT_PP(0);
	text *t_sql = PG_GETARG_TEXT_PP(1);
	char *relname = text_to_cstring(t_relname);
	char *sql = text_to_cstring(t_sql);
	List *parsetree_list;
	RawStmt *parsetree;
	Query *query;
	QueryCompletion qc;
	List *names = NIL;
	List *colNames = NIL;

	ParseState *pstate = make_parsestate(NULL);
	CreateTableAsStmt *ctas;
	StringInfoData command_buf;

	parseNameAndColumns(relname, &names, &colNames);

	initStringInfo(&command_buf);
	appendStringInfo(&command_buf, "SELECT create_immv('%s' AS '%s');", relname, sql);
	appendStringInfo(&command_buf, "%s;", sql);
	pstate->p_sourcetext = command_buf.data;

	parsetree_list = pg_parse_query(sql);

	/* XXX: should we check t_sql before command_buf? */
	if (list_length(parsetree_list) != 1)
		elog(ERROR, "invalid view definition");

	parsetree = linitial_node(RawStmt, parsetree_list);

	/* view definition should spcify SELECT query */
	if (!IsA(parsetree->stmt, SelectStmt))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("view definition must specify SELECT statement")));

	ctas = makeNode(CreateTableAsStmt);
	ctas->query = parsetree->stmt;
#if defined(PG_VERSION_NUM) && (PG_VERSION_NUM >= 140000)
	ctas->objtype = OBJECT_MATVIEW;
#else
	ctas->relkind = OBJECT_MATVIEW;
#endif
	ctas->is_select_into = false;
	ctas->into = makeNode(IntoClause);
	ctas->into->rel = makeRangeVarFromNameList(names);
	ctas->into->colNames = colNames;
	ctas->into->accessMethod = NULL;
	ctas->into->options = NIL;
	ctas->into->onCommit = ONCOMMIT_NOOP;
	ctas->into->tableSpaceName = NULL;
	ctas->into->viewQuery = parsetree->stmt;
	ctas->into->skipData = false;

	query = transformStmt(pstate, (Node *) ctas);
	Assert(query->commandType == CMD_UTILITY && IsA(query->utilityStmt, CreateTableAsStmt));

	ExecCreateImmv(pstate, (CreateTableAsStmt *) query->utilityStmt, NULL, NULL, &qc);

	PG_RETURN_INT64(qc.nprocessed);
}

/*
 * User interface for refreshing an IMMV
 */
Datum
refresh_immv(PG_FUNCTION_ARGS)
{
	text *t_relname = PG_GETARG_TEXT_PP(0);
	bool ispopulated = PG_GETARG_BOOL(1);
	char *relname = text_to_cstring(t_relname);
	QueryCompletion qc;
	StringInfoData command_buf;

	initStringInfo(&command_buf);
	appendStringInfo(&command_buf,
					 "SELECT refresh_immv('%s, %s);",
					 relname,
					 ispopulated ? "true" : "false");

	ExecRefreshImmv(makeRangeVarFromNameList(textToQualifiedNameList(t_relname)),
					!ispopulated,
					command_buf.data,
					&qc);

	PG_RETURN_INT64(qc.nprocessed);
}

/*
 * Trigger function to prevent IMMV from being changed
 */
Datum
IVM_prevent_immv_change(PG_FUNCTION_ARGS)
{
	TriggerData *trigdata = (TriggerData *) fcinfo->context;
	Relation rel = trigdata->tg_relation;

	if (!ImmvIncrementalMaintenanceIsEnabled())
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("cannot change materialized view \"%s\"", RelationGetRelationName(rel))));

	return PointerGetDatum(NULL);
}

/*
 * Create triggers to prevent IMMV from being changed
 */
void
CreateChangePreventTrigger(Oid matviewOid)
{
	ObjectAddress refaddr;
	ObjectAddress address;
	CreateTrigStmt *ivm_trigger;

	int16 types[4] = {
		TRIGGER_TYPE_INSERT, TRIGGER_TYPE_DELETE, TRIGGER_TYPE_UPDATE, TRIGGER_TYPE_TRUNCATE
	};
	int i;

	refaddr.classId = RelationRelationId;
	refaddr.objectId = matviewOid;
	refaddr.objectSubId = 0;

	ivm_trigger = makeNode(CreateTrigStmt);
	ivm_trigger->relation = NULL;
	ivm_trigger->row = false;

	ivm_trigger->timing = TRIGGER_TYPE_BEFORE;
	ivm_trigger->trigname = "IVM_prevent_immv_change";
	ivm_trigger->funcname = SystemFuncName("IVM_prevent_immv_change");
	ivm_trigger->columns = NIL;
	ivm_trigger->transitionRels = NIL;
	ivm_trigger->whenClause = NULL;
	ivm_trigger->isconstraint = false;
	ivm_trigger->deferrable = false;
	ivm_trigger->initdeferred = false;
	ivm_trigger->constrrel = NULL;
	ivm_trigger->args = NIL;

	for (i = 0; i < 4; i++)
	{
		ivm_trigger->events = types[i];
		address = CreateTrigger(ivm_trigger,
								NULL,
								matviewOid,
								InvalidOid,
								InvalidOid,
								InvalidOid,
								InvalidOid,
								InvalidOid,
								NULL,
								true,
								false);

		recordDependencyOn(&address, &refaddr, DEPENDENCY_AUTO);
	}

	/* Make changes-so-far visible */
	CommandCounterIncrement();
}

/*
 * Get relid of pg_ivm_immv
 */
Oid
PgIvmImmvRelationId(void)
{
	if (!OidIsValid(pg_ivm_immv_id))
		pg_ivm_immv_id = get_relname_relid("pg_ivm_immv", PG_CATALOG_NAMESPACE);

	return pg_ivm_immv_id;
}

/*
 * Get relid of pg_ivm_immv's primary key
 */
Oid
PgIvmImmvPrimaryKeyIndexId(void)
{
	if (!OidIsValid(pg_ivm_immv_pkey_id))
		pg_ivm_immv_pkey_id = get_relname_relid("pg_ivm_immv_pkey", PG_CATALOG_NAMESPACE);

	return pg_ivm_immv_pkey_id;
}

/*
 * Return the SELECT part of a IMMV
 */
Datum
get_immv_def(PG_FUNCTION_ARGS)
{
	Oid matviewOid = PG_GETARG_OID(0);
	Relation matviewRel = NULL;
	Query *query = NULL;
	char *querystring = NULL;

	/* Make sure IMMV is a table. */
	if (get_rel_relkind(matviewOid) != RELKIND_RELATION)
		PG_RETURN_NULL();

	matviewRel = table_open(matviewOid, AccessShareLock);
	query = get_immv_query(matviewRel);
	if (query == NULL)
	{
		table_close(matviewRel, NoLock);
		PG_RETURN_NULL();
	}

	querystring = pg_ivm_get_viewdef(matviewRel, false);

	table_close(matviewRel, NoLock);
	PG_RETURN_TEXT_P(cstring_to_text(querystring));
}

/*
 * object_access_hook function for dropping an IMMV
 */
static void
PgIvmObjectAccessHook(ObjectAccessType access, Oid classId, Oid objectId, int subId, void *arg)
{
	if (PrevObjectAccessHook)
		PrevObjectAccessHook(access, classId, objectId, subId, arg);

	if (access == OAT_DROP && classId == RelationRelationId && !OidIsValid(subId))
	{
		Relation pgIvmImmv;
		SysScanDesc scan;
		ScanKeyData key;
		HeapTuple tup;
		Oid pgIvmImmvOid = PgIvmImmvRelationId();

		if (pgIvmImmvOid == InvalidOid)
			return;

		pgIvmImmv = table_open(pgIvmImmvOid, AccessShareLock);

		ScanKeyInit(&key,
					Anum_pg_ivm_immv_immvrelid,
					BTEqualStrategyNumber,
					F_OIDEQ,
					ObjectIdGetDatum(objectId));
		scan = systable_beginscan(pgIvmImmv, PgIvmImmvPrimaryKeyIndexId(), true, NULL, 1, &key);

		tup = systable_getnext(scan);

		if (HeapTupleIsValid(tup))
			CatalogTupleDelete(pgIvmImmv, &tup->t_self);

		systable_endscan(scan);
		table_close(pgIvmImmv, NoLock);
	}
}

/*
 * isImmv
 *
 * Check if this is a IMMV from oid.
 */
bool
isImmv(Oid immv_oid)
{
	Relation pgIvmImmv = table_open(PgIvmImmvRelationId(), AccessShareLock);
	SysScanDesc scan;
	ScanKeyData key;
	HeapTuple tup;

	ScanKeyInit(&key,
				Anum_pg_ivm_immv_immvrelid,
				BTEqualStrategyNumber,
				F_OIDEQ,
				ObjectIdGetDatum(immv_oid));
	scan = systable_beginscan(pgIvmImmv, PgIvmImmvPrimaryKeyIndexId(), true, NULL, 1, &key);
	tup = systable_getnext(scan);

	systable_endscan(scan);
	table_close(pgIvmImmv, NoLock);

	if (!HeapTupleIsValid(tup))
		return false;
	else
		return true;
}

static void
pg_hook_shmem_request(void)
{
	if (PrevShmemRequestHook)
		PrevShmemRequestHook();

	RequestAddinShmemSpace(SEGMENT_SIZE + HASH_TABLE_SIZE);
	RequestNamedLWLockTranche("pg_hook", 1);
}

static void
pg_hook_shmem_startup(void)
{
	HASHCTL info;
	HASHCTL lockInfo;
	bool found = false;

	if (PrevShmemStartupHook)
		PrevShmemStartupHook();

	memset(&info, 0, sizeof(info));
	info.keysize = sizeof(QueryTableKey);
	info.entrysize = sizeof(QueryTableEntry);

	memset(&lockInfo, 0, sizeof(lockInfo));
	lockInfo.keysize = sizeof(LockedTableKey);
	lockInfo.entrysize = sizeof(LockedTableEntry);

	/*Initialize the hashtable in shared memory*/
	queryHashTable =
		ShmemInitHash("QueryTable", MAX_QUERY_NUM, MAX_QUERY_NUM, &info, HASH_ELEM | HASH_BLOBS);

	lockedRelations = ShmemInitHash("LockedRelations",32, 32, &lockInfo,
									HASH_ELEM | HASH_BLOBS);

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);

	schedule_state = ShmemInitStruct("pg_hook", SEGMENT_SIZE, &found);

	if (!found)
	{
		/*Fisrt time through, initialize data structures*/
		schedule_state->lock = &(GetNamedLWLockTranche("pg_hook")->lock);
	}

	LWLockRelease(AddinShmemInitLock);
}

static PlannedStmt *
pg_hook_planner(Query *parse, const char *query_string, int cursor_options,
				ParamListInfo bound_params)
{
	if (PrevPlanHook)
		return PrevPlanHook(parse, query_string, cursor_options, bound_params);

	return standard_planner(parse, query_string, cursor_options, bound_params);
}

void
pg_hook_execution_start(QueryDesc *queryDesc, int eflags)
{
	int status, i, j, iter;
	int running;
	QueryTableEntry *query_entry;
	RefedImmv *refed_immv;
	LOCKTAG tag;
	Bitmapset *newlyLocked = NULL;
	StringInfoData info;
	bool found;
	env_features env;


	if (PrevExecutionStartHook)
		PrevExecutionStartHook(queryDesc, eflags);
	else
		standard_ExecutorStart(queryDesc, eflags);

	/* We only schedule top-level query, if a top-level query is allowed,
	 *  sub-queries will also get executed.
	 */
	if (strcmp(queryDesc->sourceText, "") == 0 ||
		//(eflags & (EXEC_FLAG_EXPLAIN_GENERIC | EXEC_FLAG_EXPLAIN_ONLY)) ||
		(eflags & EXEC_FLAG_EXPLAIN_ONLY) || !enable_enforce(nesting_level))
		return;

	full_process++;

	LWLockAcquire(schedule_state->lock, LW_EXCLUSIVE);

	if (schedule_state->querynum >= MAX_QUERY_NUM)
		ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY), errmsg("Too many queries in the system")));

	query_entry = GetEntry(queryHashTable, queryDesc,HASH_ENTER, &found);

	if (found)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DUPLICATE_OBJECT),
				 errmsg("Found duplicate entry key, query_string: %s, previous query_string: %s",
						queryDesc->sourceText,
						query_entry->key.query_string)));
	}

	LogQuery(schedule_state, queryDesc, query_entry);

	env.wakeup_decision_id = -1;
	memcpy(env.LRU, schedule_state->tableAccessCounter, sizeof(int) * WORKING_TABLES);
	/*Trigger the reschedule*/
	Reschedule(queryHashTable, schedule_state, INCOMING_QUERY, &env);

	LWLockRelease(schedule_state->lock);

	HOLD_INTERRUPTS();
waiting:
	for (;;)
	{

		LWLockAcquire(schedule_state->lock, LW_SHARED);
		status = query_entry->status;
		running = schedule_state->runningQuery;
		LWLockRelease(schedule_state->lock);

		if (INTERRUPTS_PENDING_CONDITION())
		{
			LWLockAcquire(schedule_state->lock, LW_EXCLUSIVE);
			status = query_entry->status;
			if (status == QUERY_AVAILABLE)
			{
				nesting_level++;
				finishRunningQuery(queryDesc);
			}else if (status == QUERY_BLOCKED || status == QUERY_GIVE_UP){
				finishUnstartedQuery(queryDesc);
			}else{
				ereport(ERRCODE_SYSTEM_ERROR, errmsg("Unexpected query status: %d", status));
			}
			LWLockRelease(schedule_state->lock);
			RESUME_INTERRUPTS();
			ProcessInterrupts();

			//Should not reach here
			ereport(ERRCODE_SYSTEM_ERROR, errmsg("System did not crash after interrupt"));
		}

		/* Check for any interruptions, such as client disconnected */

		if (status == QUERY_AVAILABLE)
			break;


		/*
		 * Check if current environment is a "dead scenario"
		 * "dead scenario" means no query is running, but there are still queries waiting.
		 * If so, We trigger a rescheduling to wake a query up.
		 */
		if (running == 0)
		{
			LWLockAcquire(schedule_state->lock, LW_EXCLUSIVE);
			env.wakeup_decision_id = -1;
			memcpy(env.LRU, schedule_state->tableAccessCounter, sizeof(int) * WORKING_TABLES);
			Reschedule(queryHashTable, schedule_state, DEAD_WAKEUP, &env);
			LWLockRelease(schedule_state->lock);
		}

		pg_usleep(30);
	}

	LWLockAcquire(schedule_state->lock, LW_EXCLUSIVE);
	/* Update table access counter, as an environment feature*/
	log_table_access(schedule_state, query_entry->affected_tables);
	query_entry->start_time = clock();
	LWLockRelease(schedule_state->lock);
	RESUME_INTERRUPTS();
}

void
finishUnstartedQuery(QueryDesc *queryDesc)
{
	QueryTableEntry *query;
	bool found;
	env_features env;

	elog(LOG, "Unstarted Query: nesting_level: %d, full_process: %d", nesting_level, full_process);
	if (!(strcmp(queryDesc->sourceText, "") == 0) && enable_enforce(nesting_level) && full_process)
	{
		full_process--;
		Assert(full_process >= 0);
		query = GetEntry(queryHashTable, queryDesc, HASH_FIND, &found);
		if (!found || query == NULL)
		{
			elog(IVM_LOG_LEVEL, "Pid:%d: Cannot find Query:%s", MyProcPid, queryDesc->sourceText);
			return;
		}

		env.wakeup_decision_id = query->wakeup_by;
		memcpy(env.LRU, schedule_state->tableAccessCounter, sizeof(int) * WORKING_TABLES);

		GetEntry(queryHashTable, queryDesc, HASH_REMOVE, &found);
		UpdateStateForRemoving(schedule_state, queryDesc, query, false);

		Reschedule(queryHashTable, schedule_state, QUERY_FINISHED, &env);
		elog(LOG, "I'm fine to exit");
	}else{
		elog(LOG, "Not removing query from hash table");
	}
}

/*
 * Finish a running query.
 * Remove the query from the query hash table, update the status, and trigger a reschedule.
*/
void finishRunningQuery(QueryDesc *queryDesc)
{
	env_features env;
	QueryTableEntry *query;
	HASH_SEQ_STATUS status;
	QueryTableEntry *iter;
	bool found;

	nesting_level--;
	Assert(nesting_level >= 0);
	if (!(strcmp(queryDesc->sourceText, "") == 0) && enable_enforce(nesting_level) && full_process)
	{
		full_process--;
		Assert(full_process >= 0);

		query = GetEntry(queryHashTable, queryDesc, HASH_FIND, &found);

		if (!found || query == NULL)
		{
			elog(IVM_LOG_LEVEL, "Pid:%d: Cannot find Query:%s", MyProcPid, queryDesc->sourceText);
			return;
		}

		env.wakeup_decision_id = query->wakeup_by;
		memcpy(env.LRU, schedule_state->tableAccessCounter, sizeof(int) * WORKING_TABLES);


		query = GetEntry(queryHashTable, queryDesc, HASH_REMOVE, &found);
		UpdateStateForRemoving(schedule_state, queryDesc, query, true);


		hash_seq_init(&status, queryHashTable);

		while ((iter = (QueryTableEntry *) hash_seq_search(&status)) != NULL)
		{
			if (iter->status == QUERY_AVAILABLE){
				iter->outside_finished++;
			}
		}

		Reschedule(queryHashTable, schedule_state, QUERY_FINISHED, &env);
	}else{
		elog(LOG, "Not removing query from hash table");
	}
}

void
pg_hook_executor_run(QueryDesc *queryDesc, ScanDirection direction, uint64 count, bool execute_once)
{
	nesting_level++;
	PG_TRY();
	{
		if (prevExecutorRunHook)
			prevExecutorRunHook(queryDesc, direction, count, execute_once);
		else
			standard_ExecutorRun(queryDesc, direction, count, execute_once);
	}
	PG_CATCH();
	{
		/* This make sure query will get droped if error occured (eg. wrong syntax) */
		LWLockAcquire(schedule_state->lock, LW_EXCLUSIVE);
		finishRunningQuery(queryDesc);
		LWLockRelease(schedule_state->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	LWLockAcquire(schedule_state->lock, LW_EXCLUSIVE);
	finishRunningQuery(queryDesc);
	LWLockRelease(schedule_state->lock);
}

void
getLocksHeldByMe(StringInfo info)
{
	int i;
	LOCKTAG tag;
	initStringInfo(info);
	for (i = 0; i < immv_count; i++)
	{
		SetLocktagRelationOid(&tag, all_immvs[i]);
		if (LockHeldByMe(&tag, ExclusiveLock))
		{
			appendStringInfo(info, "%d ", all_immvs[i]);
		}
	}
}

void
pg_hook_execution_finish(QueryDesc *queryDesc)
{
	elog(LOG,"execution finish: %s", queryDesc->sourceText);
	if (PrevExecutionFinishHook)
		PrevExecutionFinishHook(queryDesc);
	else
		standard_ExecutorFinish(queryDesc);
}

void
pg_hook_executor_end(QueryDesc *queryDesc)
{
	elog(LOG,"executor end: %s", queryDesc->sourceText);
	if (PrevExecutionEndHook)
		PrevExecutionEndHook(queryDesc);
	else
		standard_ExecutorEnd(queryDesc);
}

void
pg_hook_process_utility(PlannedStmt *pstmt, const char *queryString, bool readOnlyTree,
						ProcessUtilityContext context, ParamListInfo params,
						QueryEnvironment *queryEnv, DestReceiver *dest, QueryCompletion *qc)
{
	isUtility = true;
	PG_TRY();
	{
		if (prev_ProcessUtility)
			prev_ProcessUtility(pstmt,
								queryString,
								readOnlyTree,
								context,
								params,
								queryEnv,
								dest,
								qc);
		else
			standard_ProcessUtility(pstmt,
									queryString,
									readOnlyTree,
									context,
									params,
									queryEnv,
									dest,
									qc);
	}
	PG_FINALLY();
	{
		isUtility = false;
	}
	PG_END_TRY();
}
