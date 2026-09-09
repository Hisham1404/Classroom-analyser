/**
 * Loads `results/*.json` and prepares it for display.
 *
 * The pipeline has already done every expensive thing — transcription, diarization,
 * metrics, the confidence gate. This app is a reader. That is deliberate: it means the
 * page opens instantly, needs no backend, and deploys as static files anywhere.
 *
 * One rule shapes most of this file: **a withheld metric must never render as zero.**
 * The pipeline goes to considerable trouble to refuse a number it cannot stand behind,
 * and a UI that prints "0%" would throw that away.
 */

import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, forkJoin, of } from 'rxjs';
import { catchError, map, switchMap } from 'rxjs/operators';

import { Metric, SessionResult } from '../models/session-result';

export interface TimelineBar {
  widthPercent: number;
  cssClass: string;
  label: string;
  start: number;
  end: number;
  text?: string;
}

/** Units whose value is a whole thing counted, not a measurement on a scale. */
const COUNT_UNITS = new Set(['questions', 'responses']);

/**
 * What a metric key absent from the file renders as.
 *
 * Results written before a metric existed simply have no key for it, and the page reads
 * `.formula` and friends to fill the detail panel - so an absent key threw rather than
 * rendering. That window is real: the app ships the moment it builds, while the JSON only
 * changes when the pipeline is re-run over the audio, which takes hours.
 *
 * It renders as withheld because that is what it is - no number, and a reason why.
 */
const MISSING_METRIC: Metric = {
  value: null,
  unit: null,
  band: 'unknown',
  benchmark: null,
  confidence: 0,
  formula: 'not in this file',
  explanation:
    'This recording was processed before this metric existed, so the results file ' +
    'carries no value for it.',
  interpretation: 'Re-running the pipeline over this session will publish it.',
};

@Injectable({ providedIn: 'root' })
export class SessionService {
  private readonly http = inject(HttpClient);

  /** Every session, newest first. A file that fails is skipped, not fatal. */
  load(): Observable<SessionResult[]> {
    return this.http.get<string[]>('results/index.json').pipe(
      catchError(() => of([] as string[])),
      switchMap((ids) => {
        if (!ids.length) {
          return of([] as SessionResult[]);
        }
        return forkJoin(
          ids.map((id) =>
            this.http.get<SessionResult>(`results/${id}.json`).pipe(
              catchError(() => of(null)),
            ),
          ),
        ).pipe(map((all) => all.filter((r): r is SessionResult => r !== null)));
      }),
    );
  }

  /** Human-readable metric value. `—` means withheld, and only withheld. */
  formatMetric(key: string, metric: Metric): string {
    if (metric?.value === null || metric?.value === undefined) {
      return '—';
    }
    const value = metric.value;

    if (metric.unit === 'ratio') {
      return `${Math.round(value * 100)}%`;
    }
    if (metric.unit === 'sec') {
      if (value < 60) {
        return `${value.toFixed(1)}s`;
      }
      const minutes = Math.floor(value / 60);
      const seconds = Math.round(value % 60);
      return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
    }
    if (COUNT_UNITS.has(metric.unit ?? '')) {
      return String(Math.round(value));
    }
    return value.toFixed(value < 10 ? 1 : 0);
  }

  /** The metric under `key`, or a withheld stand-in when the file predates it. */
  metricFor(result: SessionResult, key: string): Metric {
    return result.metrics?.[key] ?? MISSING_METRIC;
  }

  /**
   * Whether a metric's bar means anything.
   *
   * The gauge scales against a fixed ceiling, which works for a ratio (1.0) and for a
   * duration (10 minutes) and not at all for a count: 114 questions would peg a full
   * bar and 3 would look like a rounding error, neither of them saying anything. The
   * comparable number for a count is its rate, and that is in the interpretation text.
   */
  showsGauge(metric: Metric): boolean {
    if (metric?.value === null || metric?.value === undefined) {
      return false;
    }
    return !COUNT_UNITS.has(metric.unit ?? '');
  }

  /** Keys the pipeline refused to report — shown as such, with the reason. */
  withheldMetrics(result: SessionResult): string[] {
    return Object.entries(result.metrics)
      .filter(([, m]) => m.value === null || m.value === undefined)
      .map(([key]) => key);
  }

  showsMetrics(result: SessionResult): boolean {
    return result.confidence.verdict !== 'unreliable';
  }

  /** Proportional bars for the timeline strip — the one view that works even when
   * every word in the transcript is wrong. */
  timelineBars(result: SessionResult): TimelineBar[] {
    // The analysed window, not the whole file. A 4-minute window of a 67-minute
    // recording rendered against the file duration squeezes the strip into 6% of its
    // width and makes the axis read 0:00-67:42 over four minutes of data.
    const total = result.audio.analysed_sec || result.audio.actual_sec || 1;

    return result.timeline.map((segment) => {
      const cssClass =
        segment.kind === 'speech' ? (segment.role ?? 'unattributed') : segment.kind;

      const label =
        segment.kind === 'speech'
          ? (segment.role ?? 'speech, speaker unknown')
          : segment.kind === 'handson'
            ? 'hands-on'
            : 'quiet';

      return {
        widthPercent: ((segment.t1 - segment.t0) / total) * 100,
        cssClass,
        label,
        start: segment.t0,
        end: segment.t1,
        text: segment.text,
      };
    });
  }

  /** Same teacher over time — the "am I improving?" view. */
  groupByTeacher(results: SessionResult[]): Map<string, SessionResult[]> {
    const grouped = new Map<string, SessionResult[]>();
    for (const result of results) {
      const key = result.teacher.alias;
      grouped.set(key, [...(grouped.get(key) ?? []), result]);
    }
    return grouped;
  }

  /**
   * Capture problems worth surfacing to the M&E team. The truncation one is a real bug in
   * their recording app and matters more to them than any single lesson metric.
   */
  dataQualityNotes(result: SessionResult): string[] {
    const notes: string[] = [];
    const completeness = result.audio.completeness;

    if (result.flags.includes('audio_truncated') && completeness !== null) {
      notes.push(
        `Recording is truncated — only ${Math.round(completeness * 100)}% of the ` +
          `declared duration was saved.`,
      );
    }
    if (result.flags.includes('no_gps')) {
      notes.push('No GPS recorded for this session.');
    }
    if (result.flags.includes('roster_mismatch')) {
      notes.push('Headcount does not add up: boys + girls ≠ total.');
    }
    return notes;
  }
}
