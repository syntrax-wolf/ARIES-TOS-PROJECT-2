export interface LeaderboardEntry {
  id: string;
  timestamp: number;
  algorithm: string;
  datasetShape: string;
  accuracy: number;
  macroF1: number;
  depth: number;
  leafCount: number;
  summary: string; // short human-readable hyperparameter summary
}

const STORAGE_KEY = "sylva:leaderboard";

export function getLeaderboard(): LeaderboardEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    return [];
  }
}

function saveLeaderboard(entries: LeaderboardEntry[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // storage unavailable (private mode, quota) — fail silently, leaderboard is a nice-to-have
  }
}

export function addLeaderboardEntry(entry: LeaderboardEntry): LeaderboardEntry[] {
  const entries = getLeaderboard();
  entries.push(entry);
  entries.sort((a, b) => b.accuracy - a.accuracy || b.macroF1 - a.macroF1);
  saveLeaderboard(entries);
  return entries;
}

export function rankOf(entries: LeaderboardEntry[], id: string): number {
  const idx = entries.findIndex((e) => e.id === id);
  return idx === -1 ? entries.length : idx + 1;
}

export function clearLeaderboard() {
  saveLeaderboard([]);
}
