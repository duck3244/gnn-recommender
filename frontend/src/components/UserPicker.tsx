import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type UserOut } from "../api/client";

interface Props {
  selected: UserOut | null;
  onSelect: (u: UserOut | null) => void;
}

export function UserPicker({ selected, onSelect }: Props) {
  const [q, setQ] = useState("");
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["users", q],
    queryFn: () => api.listUsers(50, 0, q || undefined),
  });

  const onRandom = async () => {
    try {
      const u = await api.randomUser();
      onSelect(u);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search user (by original ID prefix)"
          className="flex-1 rounded border border-slate-300 px-3 py-2 text-sm"
        />
        <button
          onClick={onRandom}
          className="rounded bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700"
        >
          Random
        </button>
      </div>
      <div className="max-h-72 overflow-y-auto rounded border border-slate-200 bg-white">
        {isLoading && <div className="p-3 text-sm text-slate-500">Loading…</div>}
        {isError && (
          <div className="p-3 text-sm text-red-600">
            {(error as Error).message}
          </div>
        )}
        {!isLoading && !isError && data && (
          <ul>
            {data.items.map((u) => {
              const active = selected?.user_idx === u.user_idx;
              return (
                <li key={u.user_idx}>
                  <button
                    onClick={() => onSelect(u)}
                    className={`block w-full px-3 py-2 text-left text-sm hover:bg-slate-50 ${
                      active ? "bg-indigo-50 font-medium text-indigo-700" : ""
                    }`}
                  >
                    <span className="font-mono">#{u.user_idx}</span>{" "}
                    <span className="text-slate-500">{u.original_id.slice(0, 12)}…</span>{" "}
                    <span className="text-xs text-slate-400">
                      ({u.history_size} items)
                    </span>
                  </button>
                </li>
              );
            })}
            {data.items.length === 0 && (
              <li className="p-3 text-sm text-slate-500">No users match.</li>
            )}
          </ul>
        )}
      </div>
      {data && (
        <div className="text-xs text-slate-500">
          Showing {data.items.length} of {data.total.toLocaleString()} users
        </div>
      )}
    </div>
  );
}
