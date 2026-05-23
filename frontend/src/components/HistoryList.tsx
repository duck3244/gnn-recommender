import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function HistoryList({ userIdx }: { userIdx: number | null }) {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["history", userIdx],
    queryFn: () => api.getHistory(userIdx!, 50),
    enabled: userIdx !== null,
  });

  if (userIdx === null) {
    return <Placeholder text="Select a user to view their purchase history." />;
  }
  if (isLoading) return <Placeholder text="Loading history…" />;
  if (isError) return <Placeholder text={(error as Error).message} tone="error" />;
  if (!data || data.length === 0) {
    return <Placeholder text="This user has no recorded history." />;
  }
  return (
    <ul className="divide-y divide-slate-200">
      {data.map((it) => (
        <li key={it.item_idx} className="py-2">
          <div className="text-sm font-medium text-slate-800">{it.title}</div>
          <div className="text-xs text-slate-500 font-mono">{it.asin}</div>
        </li>
      ))}
    </ul>
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
