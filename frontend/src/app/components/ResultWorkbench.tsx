'use client';

import type { ReactNode } from 'react';
import styles from './ResultWorkbench.module.css';

/**
 * Shared Results-Workbench primitives for the three analysis surfaces
 * (Mission report, Experiment detail/compare, Convergence graph).
 *
 * Deliberately small: a section shell, label/value rows, and feedback
 * notices. Domain tones, badges, forms, cards, tables, and SVG stay
 * local to their surfaces.
 */

export function WorkbenchSection({
  title,
  meta,
  children,
}: {
  title: string;
  meta?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className={styles.section} aria-label={title}>
      <div className={styles.header}>
        <h2>{title}</h2>
        {meta}
      </div>
      {children}
    </section>
  );
}

export interface ResultRow {
  key: string;
  label: ReactNode;
  value: ReactNode;
}

export function ResultRows({ rows }: { rows: ResultRow[] }) {
  return (
    <div className={styles.grid}>
      {rows.map((row) => (
        <div className={styles.row} key={row.key}>
          <span>{row.label}</span>
          <span>{row.value}</span>
        </div>
      ))}
    </div>
  );
}

export function WorkbenchNotice({
  tone,
  align = 'start',
  children,
}: {
  tone: 'error' | 'loading' | 'empty';
  align?: 'start' | 'center';
  children: ReactNode;
}) {
  const toneClass =
    tone === 'error'
      ? styles.noticeError
      : tone === 'loading'
        ? styles.noticeLoading
        : styles.noticeEmpty;
  return (
    <div
      className={`${styles.notice} ${toneClass} ${
        align === 'center' ? styles.noticeCenter : ''
      }`}
      role={tone === 'error' ? 'alert' : 'status'}
    >
      {children}
    </div>
  );
}
