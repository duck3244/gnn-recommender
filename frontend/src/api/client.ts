// Thin fetch wrapper. All requests go to /api/* which Vite proxies to FastAPI
// in dev and which FastAPI handles directly in prod (single-port).

const BASE = "/api";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export interface UserOut {
  user_idx: number;
  original_id: string;
  history_size: number;
}

export interface ItemOut {
  item_idx: number;
  asin: string;
  title: string;
}

export interface RecommendationOut extends ItemOut {
  score: number;
}

export interface UserListOut {
  total: number;
  items: UserOut[];
}

export interface Health {
  status: string;
  model_loaded: boolean;
  num_users: number;
  num_items: number;
}

export const api = {
  health: () => http<Health>("/health"),
  listUsers: (limit = 100, offset = 0, q?: string) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (q) params.set("q", q);
    return http<UserListOut>(`/users?${params}`);
  },
  randomUser: () => http<UserOut>("/users/random"),
  getUser: (idx: number) => http<UserOut>(`/users/${idx}`),
  getHistory: (idx: number, limit = 50) =>
    http<ItemOut[]>(`/users/${idx}/history?limit=${limit}`),
  getRecommendations: (idx: number, k = 10) =>
    http<RecommendationOut[]>(`/users/${idx}/recommendations?k=${k}`),
};
