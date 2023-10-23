#include "postgres.h"
#include "storage/lwlock.h"
#include "utils/elog.h"
#include "stdlib.h"

#include "pg_ivm.h"

static void Reschedule(ScheduleState *state);

void
InsertQuery(ScheduleState *state, const char *query)
{
  int index;

  index = state->querynum;

  if (index >= MAX_QUERY_NUM)
    {
      ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Query number exceeds the maximum number of queries can be saved!")));
    }

  state->querynum++;
  strcpy(state->queryTable.queries[index], query);

  Reschedule(state);

  elog(INFO, "Inserting query: %s", query);
}



/* TODO: Implement a heuristic based rescheduling algorithm*/
static void
Reschedule(ScheduleState *state)
{
  elog(INFO, "Rescheduling");
}
