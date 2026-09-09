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

import {
  METRIC_LABELS,
  METRIC_ORDER,
  Metric,
  SessionResult,
} from '../models/session-result';
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

describe('SessionService counts', () => {
  let service: SessionService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [SessionService, provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(SessionService);
  });

  // A count is the one metric shape where the existing formatter was actively wrong:
  // it falls through to `toFixed(value < 10 ? 1 : 0)`, so five questions rendered as
  // "5.0". Nothing is ever five point zero questions.
  it('formats a question count as a whole number', () => {
    expect(service.formatMetric('M7_teacher_questions', metric(5, { unit: 'questions' })))
      .toBe('5');
  });

  it('formats a large count without a decimal point', () => {
    expect(service.formatMetric('M7_teacher_questions', metric(114, { unit: 'questions' })))
      .toBe('114');
  });

  it('shows zero questions as zero, not as withheld', () => {
    // A lesson with no questions in it is a finding. The dash means "could not
    // measure", and the two must never print the same.
    expect(service.formatMetric('M7_teacher_questions', metric(0, { unit: 'questions' })))
      .toBe('0');
  });

  it('still shows a withheld count as withheld', () => {
    expect(service.formatMetric('M8_student_responses', metric(null, { unit: 'responses' })))
      .toBe('—');
  });

  // The gauge scales against a fixed ceiling (10 for anything that is not a ratio or a
  // duration). A count has no ceiling - 114 questions would peg a full bar and 3 would
  // look like a rounding error, both of them meaning nothing. The rate that IS
  // comparable lives in the interpretation text instead.
  it('draws no gauge for a count', () => {
    expect(service.showsGauge(metric(114, { unit: 'questions' }))).toBe(false);
    expect(service.showsGauge(metric(2, { unit: 'responses' }))).toBe(false);
  });

  it('still draws a gauge for the metrics that have a ceiling', () => {
    expect(service.showsGauge(metric(0.77))).toBe(true);
    expect(service.showsGauge(metric(412, { unit: 'sec' }))).toBe(true);
  });

  it('draws no gauge for a withheld metric', () => {
    expect(service.showsGauge(metric(null))).toBe(false);
  });

  it('lists both counts among the metrics a teacher reads', () => {
    // The brief asks for these two by name, so they must actually reach the page -
    // a metric absent from METRIC_ORDER is in the JSON and nowhere else.
    expect(METRIC_ORDER).toContain('M7_teacher_questions');
    expect(METRIC_ORDER).toContain('M8_student_responses');
  });

  it('gives every listed metric a label', () => {
    for (const key of METRIC_ORDER) {
      expect(METRIC_LABELS[key]).toBeTruthy();
    }
  });
});

describe('SessionService missing metrics', () => {
  let service: SessionService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [SessionService, provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(SessionService);
  });

  // A results file written before a metric existed simply has no key for it. The page
  // reads `metrics[key].formula` to fill the detail panel, so an absent key threw
  // instead of rendering - and the window where that happens is real: the app deploys
  // the moment it builds, while the JSON only changes when the pipeline is re-run.
  it('treats a metric missing from an older results file as withheld', () => {
    const r = makeResult();
    expect(r.metrics['M7_teacher_questions']).toBeUndefined();
    expect(service.metricFor(r, 'M7_teacher_questions').value).toBeNull();
  });

  it('gives a missing metric something to say rather than an empty panel', () => {
    const m = service.metricFor(makeResult(), 'M8_student_responses');
    expect(m.formula).toBeTruthy();
    expect(m.explanation).toBeTruthy();
    expect(m.interpretation).toBeTruthy();
  });

  it('returns the real metric when the file does carry it', () => {
    const r = makeResult();
    expect(service.metricFor(r, 'M1_teacher_talk_ratio').value).toBe(0.77);
  });

  it('does not count a missing metric as a withheld one', () => {
    // `withheldMetrics` drives the data-quality note. A key that was never written is
    // not the pipeline refusing to publish a number - it is an older file.
    expect(service.withheldMetrics(makeResult())).not.toContain('M7_teacher_questions');
  });
});
