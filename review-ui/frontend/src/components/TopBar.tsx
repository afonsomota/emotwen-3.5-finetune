interface TopBarProps {
  currentPosition: number;
  totalCount: number;
  sourceFilter: string | null;
  sources: string[];
  stats: { good: number; okay: number; bad: number };
  statusFilter: "reviewed" | "unreviewed" | null;
  onSourceChange: (source: string | null) => void;
  onStatusToggle: () => void;
}

export default function TopBar({
  currentPosition,
  totalCount,
  sourceFilter,
  sources,
  stats,
  statusFilter,
  onSourceChange,
  onStatusToggle,
}: TopBarProps) {
  return (
    <div className="topbar">
      <span className="topbar-progress">
        {totalCount > 0 ? `${currentPosition} / ${totalCount}` : "0 / 0"}
      </span>
      <select
        className="topbar-select"
        value={sourceFilter || ""}
        onChange={(e) => onSourceChange(e.target.value || null)}
      >
        <option value="">All sources</option>
        {sources.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
      <button
        className={`status-btn${statusFilter === "unreviewed" ? " active" : ""}`}
        onClick={onStatusToggle}
      >
        {statusFilter === "unreviewed" ? "Unreviewed" : "All"}
      </button>
      <div className="topbar-stats">
        <span className="stat-chip stat-good">{stats.good} G</span>
        <span className="stat-chip stat-okay">{stats.okay} O</span>
        <span className="stat-chip stat-bad">{stats.bad} B</span>
      </div>
    </div>
  );
}
