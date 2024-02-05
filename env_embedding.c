#include "conf.h"
#include "pg_ivm.h"

/* Update the table access information. If a table has been accessed in a query
 * then its access count is set to 0, otherwise it is incremented by 1. */
void
log_table_access(ScheduleState *ss, Oid *affected_table)
{
	int i;
	bool accessed[WORKING_TABLES] = { 0 };

	for (i = 0; i < MAX_AFFECTED_TABLE && affected_table[i] != 0; i++)
	{
		accessed[getIndexForTableEmbeeding(affected_table[i])] = true;
	}

	for (i = 0; i < WORKING_TABLES; i++)
	{
		if (accessed[i])
		{
			ss->tableAccessCounter[i] = 0;
		}
		else
		{
      /* Ensure the value is within a stable range. */
			if (ss->tableAccessCounter[i] < 10)
				ss->tableAccessCounter[i]++;
		}
	}
}
