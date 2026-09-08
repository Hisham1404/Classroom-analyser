import { CommonModule } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';

import { METRIC_LABELS, METRIC_ORDER, SessionResult } from './models/session-result';
import { SessionService, TimelineBar } from './services/session.service';

type View = 'teacher' | 'trend' | 'quality';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  private readonly sessions = inject(SessionService);

  readonly all = signal<SessionResult[]>([]);
  readonly selectedId = signal<string | null>(null);
  readonly view = signal<View>('teacher');
  readonly loading = signal(true);
  readonly expandedMetric = signal<string | null>(null);
  readonly showTranscript = signal(false);

  readonly metricOrder = METRIC_ORDER;
  readonly metricLabels = METRIC_LABELS;

  readonly selected = computed(() => {
    const id = this.selectedId();
    return this.all().find((s) => s.session_id === id) ?? this.all()[0] ?? null;
  });

  readonly bars = computed<TimelineBar[]>(() => {
    const session = this.selected();
    return session ? this.sessions.timelineBars(session) : [];
  });

  readonly spokenSegments = computed(() =>
    (this.selected()?.timeline ?? []).filter((s) => s.kind === 'speech' && s.text),
  );

  readonly byTeacher = computed(() => this.sessions.groupByTeacher(this.all()));

  readonly teacherGroups = computed(() =>
    [...this.byTeacher().entries()].map(([alias, list]) => ({ alias, list })),
  );

  constructor() {
    this.sessions.load().subscribe((results) => {
      this.all.set(results);
      this.selectedId.set(results[0]?.session_id ?? null);
      this.loading.set(false);
    });
  }

  select(id: string): void {
    this.selectedId.set(id);
    this.expandedMetric.set(null);
    this.showTranscript.set(false);
  }

  toggleMetric(key: string): void {
    this.expandedMetric.set(this.expandedMetric() === key ? null : key);
  }

  metricValue(session: SessionResult, key: string): string {
    return this.sessions.formatMetric(key, session.metrics[key]);
  }

  showsMetrics(session: SessionResult): boolean {
    return this.sessions.showsMetrics(session);
  }

  qualityNotes(session: SessionResult): string[] {
    return this.sessions.dataQualityNotes(session);
  }

  /** Position of the benchmark marker on a metric card, as a percentage. */
  benchmarkOffset(session: SessionResult, key: string): number | null {
    const metric = session.metrics[key];
    if (metric?.benchmark == null || metric.value == null) {
      return null;
    }
    return Math.min(100, Math.max(0, metric.benchmark * 100));
  }

  barOffset(session: SessionResult, key: string): number {
    const metric = session.metrics[key];
    if (metric?.value == null) {
      return 0;
    }
    if (metric.unit === 'ratio') {
      return Math.min(100, metric.value * 100);
    }
    // Non-ratio metrics have no natural ceiling; scale against a sensible upper bound.
    const ceiling = metric.unit === 'sec' ? 600 : 10;
    return Math.min(100, (metric.value / ceiling) * 100);
  }

  clock(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const rest = Math.floor(seconds % 60);
    return `${minutes}:${String(rest).padStart(2, '0')}`;
  }

  totalKeys(session: SessionResult): { key: string; value: number }[] {
    return Object.entries(session.totals_sec)
      .filter(([key, value]) => value > 0 && key !== 'unattributed')
      .map(([key, value]) => ({ key, value }));
  }
}
