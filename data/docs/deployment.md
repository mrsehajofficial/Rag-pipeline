---
title: Deployment Runbook
owner: platform-team
version: 3
---

# Deployment Runbook

## Rolling a Release

We deploy with a rolling update: start new pods, wait for readiness, then drain old pods.
Never use a recreate strategy in production; it causes a hard outage.

To roll back, redeploy the previous image tag. Rollback must stay under five minutes,
so keep at least three prior images in the registry.

## Database Migrations

Run migrations as a separate job before the app rollout. Migrations must be backward
compatible with the currently running version, otherwise the rollback path breaks.

Never run a destructive migration in the same release that stops using the column. Split
it across two releases: stop writing first, then drop.

## Health Checks

The readiness probe hits `/readyz` and must return 200 only after the connection pool is
warm. The liveness probe hits `/livez` and must never depend on downstream services, or a
dependency outage will restart every pod at once.

## Rollback Triggers

Roll back when the error rate exceeds one percent for five minutes, when p99 latency
doubles, or when any data correctness alert fires. Do not debug in production.
