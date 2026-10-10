// Separate the training record from the live fleet and the last completed race.
export function summarizeRun(run) {
  const frame = run?.frame;
  const last = run?.gens?.at(-1);
  const laps = run?.config?.laps ?? 1;
  const progress = [run?.best, last?.best, frame?.hud?.progress]
    .filter((value) => Number.isFinite(value));
  const best = progress.length ? Math.max(...progress) : null;
  const total = frame?.n ?? run?.config?.population;
  const finished = frame ? frame.cars.filter((car) => car[3] === 2).length : last?.finished;
  const gen = frame?.gen ?? last?.gen;
  const count = (value) => value == null ? '—' : `${value} / ${total ?? '—'}`;
  return {
    iterationLabel: run?.learner === 'evolution' ? 'Поколение' : 'Итерация',
    iteration: gen == null ? '—' : String(gen + 1),
    best: best == null ? '—' : laps > 1
      ? `${best.toFixed(1).replace('.', ',')} / ${laps} кр.`
      : `${Math.round(best * 100)}%`,
    alive: count(frame?.alive),
    finished: count(finished),
    raceHint: frame ? 'В текущем заезде' : 'В последнем заезде',
  };
}
