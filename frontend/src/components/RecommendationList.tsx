import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

interface Props {
  userIdx: number | null;
  k: number;
}

export function RecommendationList({ userIdx, k }: Props) {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["recs", userIdx, k],
    queryFn: () => api.getRecommendations(userIdx!, k),
    enabled: userIdx !== null,
  });

  if (userIdx === null) {
    return <Placeholder text="Select a user to see Top-K recommendations." />;
  }
  if (isLoading) return <Placeholder text="Computing recommendations…" />;
  if (isError) return <Placeholder text={(error as Error).message} tone="error" />;
  if (!data || data.length === 0) {
    return <Placeholder text="No recommendations available." />;
  }
  return (
    <ol className="divide-y divide-slate-200">
      {data.map((it, i) => (
        <li key={it.item_idx} className="flex gap-3 py-2">
          <span className="w-6 text-right text-sm font-mono text-slate-400">
            {i + 1}
          </span>
          <div className="flex-1">
            <div className="text-sm font-medium text-slate-800">{it.title}</div>
            <div className="text-xs text-slate-500 font-mono">{it.asin}</div>
          </div>
          <span className="self-center rounded bg-slate-100 px-2 py-0.5 text-xs font-mono text-slate-600">
            {it.score.toFixed(3)}
          </span>
        </li>
      ))}
    </ol>
  );
}

function Placeholder({
  text,
  tone = "muted",
}: {
  text: string;
  tone?: "muted" | "error";
}) {
  const cls = tone === "error" ? "text-red-600" : "text-slate-500";
  return <div className={`py-6 text-sm ${cls}`}>{text}</div>;
}
