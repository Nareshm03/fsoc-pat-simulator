/**
 * Backend endpoint configuration.
 *
 * Overridden via environment (see frontend/.env.example):
 *   NEXT_PUBLIC_API_URL - REST base URL (default http://localhost:8000)
 *   NEXT_PUBLIC_WS_URL  - WebSocket base URL (default ws://localhost:8000)
 *
 * Local development behavior is unchanged when unset.
 */
export const API_BASE: string =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export const WS_BASE: string =
  process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000';
