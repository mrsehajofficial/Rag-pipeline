Incident Report: INC-2291 Unhandled NullReferenceError in Export Worker

Severity: SEV-2
Owner: payments-team

## Summary

Between 03:12 and 03:48 UTC the export worker crashed on every job involving a customer
with no configured currency. The worker dereferenced `customer.currency.code` without a
null check. The queue backed up to roughly forty thousand jobs.

## Root Cause

The `ExportWorker.run` method assumed every customer record had a currency. Legacy records
imported in 2023 predate the currency column being required, so roughly two percent of
customers have `currency = null`.

We had a null check in the older `SyncWorker`, but it was never carried over when the
export worker was extracted into its own service in January.

## Resolution

We added a default currency fallback and a defensive guard. Jobs drained over the
following six hours.

## Action Items

1. Add a schema constraint so currency is never null again. Owner: payments-team, due next sprint.
2. Add a null-safety linter rule for nested attribute access on ORM models.
3. Alert when any worker restart count exceeds zero in a fifteen minute window.
4. Audit the other three workers extracted in January for the same missed check.
