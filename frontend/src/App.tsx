import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type UserOut } from "./api/client";
import { UserPicker } from "./components/UserPicker";
import { HistoryList } from "./components/HistoryList";
import { RecommendationList } from "./components/RecommendationList";

export default function App() {
  const [selected, setSelected] = useState<UserOut | null>(null);
  const [k, setK] = useState(10);

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
  });

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-lg font-semibold text-slate-900">
              GNN Product Recommender
            </h1>
            <p className="text-xs text-slate-500">
              LightGCN · Amazon Beauty · single-user MVP
            </p>
          </div>
          <div className="text-right text-xs">
            <div
              className={`inline-flex items-center gap-1 rounded px-2 py-1 ${
                health?.model_loaded
                  ? "bg-green-50 text-green-700"
                  : "bg-yellow-50 text-yellow-700"
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  health?.model_loaded ? "bg-green-500" : "bg-yellow-500"
                }`}
              />
              {health
                ? health.model_loaded
                  ? `model ready · ${health.num_users.toLocaleString()} users`
                  : "model not loaded"
                : "connecting…"}
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-6">
        <section className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <div className="md:col-span-1">
            <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Pick a user
            </h2>
            <UserPicker selected={selected} onSelect={setSelected} />

            <div className="mt-4 rounded border border-slate-200 bg-white p-3">
              <label className="flex items-center justify-between text-sm">
                <span className="text-slate-700">Top-K</span>
                <span className="font-mono text-slate-900">{k}</span>
              </label>
              <input
                type="range"
                min={1}
                max={50}
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="w-full"
              />
            </div>

            {selected && (
              <div className="mt-4 rounded border border-slate-200 bg-white p-3 text-xs text-slate-600">
                <div>
                  <span className="text-slate-400">user_idx:</span>{" "}
                  <span className="font-mono">{selected.user_idx}</span>
                </div>
                <div>
                  <span className="text-slate-400">original_id:</span>{" "}
                  <span className="font-mono break-all">{selected.original_id}</span>
                </div>
                <div>
                  <span className="text-slate-400">history:</span>{" "}
                  <span className="font-mono">{selected.history_size}</span>
                </div>
              </div>
            )}
          </div>

          <div className="md:col-span-1">
            <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Purchase History
            </h2>
            <div className="rounded border border-slate-200 bg-white px-3">
              <HistoryList userIdx={selected?.user_idx ?? null} />
            </div>
          </div>

          <div className="md:col-span-1">
            <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Top-{k} Recommendations
            </h2>
            <div className="rounded border border-slate-200 bg-white px-3">
              <RecommendationList userIdx={selected?.user_idx ?? null} k={k} />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
