/**
 * The contract written by the offline pipeline (`src/pipeline.py`).
 *
 * This mirrors `results/<session_id>.json` exactly. Everything upstream — VAD, ASR
 * backend, diarization — can be rewritten freely as long as this shape holds, which is
 * what lets the pipeline change without touching the app.
 *
 * Every metric value is nullable on purpose. A metric that could not be trusted is
 * withheld rather than guessed, so `null` is a real state the UI has to render, not an
 * error to code around.
 */

export type Verdict = 'usable' | 'partial' | 'unreliable';
export type SegmentKind = 'speech' | 'handson' | 'dead';
export type Role = 'teacher' | 'student';
export type Band = 'low' | 'normal' | 'high' | 'unknown';

export interface Metric {
  value: number | null;
  unit: string | null;
  band: Band;
  benchmark: number | null;
  confidence: number;
  formula: string;
  explanation: string;
  interpretation: string;
}

export interface TimelineSegment {
  t0: number;
  t1: number;
  kind: SegmentKind;
  rms_db: number;
  role?: Role;
  role_conf?: number;
  text?: string;
}

export interface Insight {
  headline: string;
  suggestion: string;
  summary: string;
  metrics_shown: string[];
  verdict: Verdict;
  generated_by: string;
  whatsapp: string;
}

export interface SessionResult {
  schema_version: string;
  session_id: string;
  teacher: { id: string | null; alias: string };
  activity: string | null;
  recorded_at: string | null;
  roster: { total: number; boys: number; girls: number };
  audio: {
    declared_sec: number | null;
    actual_sec: number;
    /** How much was actually processed — the timeline and metrics cover this span. */
    analysed_sec: number;
    completeness: number | null;
    sample_rate: number;
    channels: number;
    has_gps: boolean;
  };
  models: {
    asr: string | null;
    asr_backend: string | null;
    attribution: string;
    vad: string;
  };
  timeline: TimelineSegment[];
  totals_sec: Record<string, number>;
  metrics: Record<string, Metric>;
  confidence: {
    overall: number;
    components: Record<string, number>;
    verdict: Verdict;
  };
  insight: Insight;
  flags: string[];
}

/** Metric keys in the order a teacher should read them. */
export const METRIC_ORDER = [
  'M1_teacher_talk_ratio',
  'M2_student_participation',
  'M3_interaction_density',
  'M4_longest_teacher_stretch',
  'M5_wait_time_1',
  'M7_teacher_questions',
  'M8_student_responses',
] as const;

export const METRIC_LABELS: Record<string, string> = {
  M1_teacher_talk_ratio: 'Teacher talk',
  M2_student_participation: 'Student participation',
  M3_interaction_density: 'Exchanges',
  M4_longest_teacher_stretch: 'Longest stretch',
  M5_wait_time_1: 'Wait time',
  M7_teacher_questions: 'Questions asked',
  M8_student_responses: 'Questions answered',
};
