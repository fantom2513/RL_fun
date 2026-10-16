const COMPLETED = new Set(['finished', 'stopped', 'error']);

export function completedIds(runs) {
  return runs.filter((run) => COMPLETED.has(run.status)).map((run) => run.id);
}

// Keep failed entries visible; a run already removed by another page is a successful deletion.
export async function removeRuns(ids, remove) {
  const removed = [];
  const failed = [];
  for (const id of new Set(ids)) {
    try {
      await remove(id);
      removed.push(id);
    } catch (error) {
      (error.status === 404 ? removed : failed).push(id);
    }
  }
  return { removed, failed };
}
