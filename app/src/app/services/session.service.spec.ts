/**
 * Loading and presenting results.
 *
 * The rules under test are the ones the whole design rests on:
 *   - a withheld metric renders as withheld, never as zero
 *   - an unreliable session shows no numbers at all
 *   - a failed fetch degrades to an empty list, not a broken page
 */

import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';

import { Metric, SessionResult } from '../models/session-result';
import { SessionService } from './session.service';

function metric(value: number | null, extra: Partial<Metric> = {}): Metric {
  return {
    value,
    unit: 'ratio',
    band: value === null ? 'unknown' : 'high',
    benchmark: 0.4,
    confidence: 0.8,
    formula: 'f',
    explanation: 'e',
    interpretation: 'i',
    ...extra,
  };
}

function makeResult(over: Partial<SessionResult> = {}): SessionResult {
  return {
    schema_version: '1.0',
    session_id: 'OD11163_2025-12-23-121239',
    teacher: { id: 'OD11163', alias: 'Teacher A' },
    activity: 'Trumpet',
    recorded_at: '2025-12-23T13:21:25',
    roster: { total: 22, boys: 15, girls: 7 },
    audio: {
      declared_sec: 4062.4, actual_sec: 4062.4, analysed_sec: 240, completeness: 1.0,
      sample_rate: 16000, channels: 1, has_gps: true,
    },
    models: {
      asr: 'indicconformer', asr_backend: 'indic',
      attribution: 'pyannote-community-1', vad: 'silero@0.5',
    },
    timeline: [
      { t0: 0, t1: 100, kind: 'speech', rms_db: -20, role: 'teacher', role_conf: 0.9 },
      { t0: 100, t1: 130, kind: 'speech', rms_db: -30, role: 'student', role_conf: 0.9 },
      { t0: 130, t1: 230, kind: 'handson', rms_db: -28 },
      { t0: 230, t1: 240, kind: 'dead', rms_db: -60 },
    ],
    totals_sec: { teacher: 100, student: 30, handson: 100, dead: 10 },
    metrics: {
      M1_teacher_talk_ratio: metric(0.77),
      M2_student_participation: metric(1.2),
      M3_interaction_density: metric(0.5),
      M4_longest_teacher_stretch: metric(100),
      M5_wait_time_1: metric(null),
    },
    confidence: {
      overall: 0.91,
      components: { speech_density: 1, attribution: 0.9, completeness: 1, asr: 0.8 },
      verdict: 'usable',
    },
    insight: {
      headline: 'You spoke 77% of the talking time.',
      suggestion: 'Try breaking it with a question.',
      summary: 'Trumpet: 2m of talking, 1m hands-on.',
      metrics_shown: ['M1_teacher_talk_ratio'],
      verdict: 'usable',
      generated_by: 'rules-v1',
      whatsapp: 'नमस्ते Teacher A!',
    },
    flags: [],
    ...over,
  };
}

describe('SessionService', () => {
  let service: SessionService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [SessionService, provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(SessionService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('loads the session index and then each result', () => {
    let loaded: SessionResult[] = [];
    service.load().subscribe((r) => (loaded = r));

    http.expectOne('results/index.json').flush(['OD11163_2025-12-23-121239']);
    http.expectOne('results/OD11163_2025-12-23-121239.json').flush(makeResult());

    expect(loaded.length).toBe(1);
    expect(loaded[0].teacher.alias).toBe('Teacher A');
  });

  it('degrades to an empty list when the index cannot be fetched', () => {
    let loaded: SessionResult[] | undefined;
    service.load().subscribe((r) => (loaded = r));

    http.expectOne('results/index.json').error(new ProgressEvent('404'));
    expect(loaded).toEqual([]);
  });

  it('skips a session whose file fails rather than losing the whole page', () => {
    let loaded: SessionResult[] = [];
    service.load().subscribe((r) => (loaded = r));

    http.expectOne('results/index.json').flush(['good', 'broken']);
    http.expectOne('results/good.json').flush(makeResult({ session_id: 'good' }));
    http.expectOne('results/broken.json').error(new ProgressEvent('500'));

    expect(loaded.length).toBe(1);
    expect(loaded[0].session_id).toBe('good');
  });
});

describe('SessionService presentation', () => {
  let service: SessionService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [SessionService, provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(SessionService);
  });

  it('formats a ratio metric as a percentage', () => {
    expect(service.formatMetric('M1_teacher_talk_ratio', metric(0.77))).toBe('77%');
  });

  it('formats seconds as minutes and seconds', () => {
    const m = metric(412, { unit: 'sec' });
    expect(service.formatMetric('M4_longest_teacher_stretch', m)).toBe('6m 52s');
  });

  it('shows a withheld metric as withheld, never as zero', () => {
    expect(service.formatMetric('M5_wait_time_1', metric(null))).toBe('—');
  });

  it('distinguishes a genuine zero from a withheld metric', () => {
    const zero = metric(0, { unit: 'switches/min' });
    expect(service.formatMetric('M3_interaction_density', zero)).not.toBe('—');
  });

  it('reports which metrics were withheld', () => {
    const r = makeResult();
    expect(service.withheldMetrics(r)).toContain('M5_wait_time_1');
    expect(service.withheldMetrics(r)).not.toContain('M1_teacher_talk_ratio');
  });

  it('says an unreliable session shows no metrics', () => {
    const r = makeResult({
      confidence: { overall: 0.2, components: {}, verdict: 'unreliable' },
    });
    expect(service.showsMetrics(r)).toBeFalse();
  });

  it('says a usable session shows metrics', () => {
    expect(service.showsMetrics(makeResult())).toBeTrue();
  });

  it('builds timeline bars that span the whole recording', () => {
    const bars = service.timelineBars(makeResult());
    const total = bars.reduce((sum, b) => sum + b.widthPercent, 0);
    expect(total).toBeCloseTo(100, 1);
  });

  it('colours timeline bars by role and kind', () => {
    const bars = service.timelineBars(makeResult());
    expect(bars[0].cssClass).toBe('teacher');
    expect(bars[1].cssClass).toBe('student');
    expect(bars[2].cssClass).toBe('handson');
    expect(bars[3].cssClass).toBe('dead');
  });

  it('groups sessions by teacher so a trend can be shown', () => {
    const a = makeResult({ session_id: 'a', teacher: { id: 'T1', alias: 'Teacher A' } });
    const b = makeResult({ session_id: 'b', teacher: { id: 'T1', alias: 'Teacher A' } });
    const c = makeResult({ session_id: 'c', teacher: { id: 'T2', alias: 'Teacher B' } });

    const grouped = service.groupByTeacher([a, b, c]);
    expect(grouped.get('Teacher A')?.length).toBe(2);
    expect(grouped.get('Teacher B')?.length).toBe(1);
  });

  it('flags a truncated recording for the M&E view', () => {
    const r = makeResult({ flags: ['audio_truncated'], audio: { ...makeResult().audio, completeness: 0.34 } });
    expect(service.dataQualityNotes(r).some((n) => n.includes('34%'))).toBeTrue();
  });

  it('has no data-quality notes for a clean recording', () => {
    expect(service.dataQualityNotes(makeResult())).toEqual([]);
  });
});
