import { useState, useEffect, useRef, useCallback } from "react";
import type { Rating, Tag, Conversation, Stats } from "./types";
import { getSources, getConversations, getConversation, postFeedback, getStats } from "./api";
import useKeyboard from "./hooks/useKeyboard";
import TopBar from "./components/TopBar";
import ChatView from "./components/ChatView";
import RatingBar from "./components/RatingBar";

function App() {
  const [sources, setSources] = useState<string[]>([]);
  const [sourceFilter, setSourceFilter] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<"reviewed" | "unreviewed" | null>(null);
  const [conversationIndices, setConversationIndices] = useState<number[]>([]);
  const [cursorPos, setCursorPos] = useState(0);
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [localRating, setLocalRating] = useState<Rating | null>(null);
  const [localTags, setLocalTags] = useState<Tag[]>([]);
  const [stats, setStats] = useState<Stats>({
    total: 0,
    reviewed: 0,
    ratings: { good: 0, okay: 0, bad: 0 },
    tags: { "advice-leaking": 0, "too-long": 0, "off-topic": 0, "repetitive": 0 },
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [connectionError, setConnectionError] = useState(false);
  const [ratingCooldown, setRatingCooldown] = useState(false);
  const [isEnd, setIsEnd] = useState(false);

  // Ref to always have latest state in keyboard handlers
  const stateRef = useRef({
    conversationIndices,
    cursorPos,
    localTags,
    localRating,
    ratingCooldown,
    sourceFilter,
    statusFilter,
    currentConversation,
  });
  stateRef.current = {
    conversationIndices,
    cursorPos,
    localTags,
    localRating,
    ratingCooldown,
    sourceFilter,
    statusFilter,
    currentConversation,
  };

  // Load a conversation at a given cursor position within the current indices
  const loadConversation = useCallback(async (indices: number[], pos: number) => {
    if (indices.length === 0) {
      setCurrentConversation(null);
      return;
    }
    const idx = indices[pos];
    try {
      const conv = await getConversation(idx);
      setCurrentConversation(conv);
      if (conv.feedback) {
        setLocalRating(conv.feedback.rating);
        setLocalTags(conv.feedback.tags);
      } else {
        setLocalRating(null);
        setLocalTags([]);
      }
      setIsEnd(false);
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  // Fetch conversation list + stats for the current filters
  const fetchList = useCallback(async (source: string | null, status: string | null) => {
    try {
      const [convResp, statsResp] = await Promise.all([
        getConversations({ source, status, limit: 5000 }),
        getStats(source),
      ]);
      const indices = convResp.items.map((item) => item.idx);
      setConversationIndices(indices);
      setStats(statsResp);
      setCursorPos(0);
      await loadConversation(indices, 0);
    } catch (e: any) {
      if (e.message && e.message.includes("loading")) {
        // Dataset still loading, poll
        setTimeout(() => fetchList(source, status), 2000);
        return;
      }
      setConnectionError(true);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [loadConversation]);

  // Startup
  useEffect(() => {
    async function init() {
      try {
        const sourcesResp = await getSources();
        setSources(sourcesResp.sources);
        await fetchList(null, null);
      } catch (e: any) {
        if (e.message && e.message.includes("loading")) {
          setTimeout(() => init(), 2000);
          return;
        }
        setConnectionError(true);
        setLoading(false);
      }
    }
    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Filter changes (skip initial mount)
  const didMount = useRef(false);
  useEffect(() => {
    if (!didMount.current) {
      didMount.current = true;
      return;
    }
    setLoading(true);
    fetchList(sourceFilter, statusFilter);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceFilter, statusFilter]);

  // Handlers
  const handleRate = useCallback(async (rating: Rating) => {
    const s = stateRef.current;
    if (s.ratingCooldown) return;
    if (s.conversationIndices.length === 0) return;

    stateRef.current.ratingCooldown = true;  // sync ref immediately to prevent races
    const idx = s.conversationIndices[s.cursorPos];
    setLocalRating(rating);
    setRatingCooldown(true);

    try {
      await postFeedback({
        conversation_idx: idx,
        rating,
        tags: s.localTags,
      });
      // Refresh stats in background
      getStats(s.sourceFilter).then(setStats).catch(() => {});

      // When filtering unreviewed, re-fetch the list since the rated item
      // is no longer unreviewed and the indices have changed.
      if (s.statusFilter === "unreviewed") {
        await fetchList(s.sourceFilter, s.statusFilter);
      } else {
        // Advance
        const nextPos = s.cursorPos + 1;
        if (nextPos < s.conversationIndices.length) {
          setCursorPos(nextPos);
          await loadConversation(s.conversationIndices, nextPos);
        } else {
          setIsEnd(true);
        }
      }
    } catch (e: any) {
      setError(e.message);
      // Auto-dismiss after 3s
      setTimeout(() => setError(null), 3000);
    } finally {
      setTimeout(() => {
        stateRef.current.ratingCooldown = false;
        setRatingCooldown(false);
      }, 300);
    }
  }, [loadConversation, fetchList]);

  const handleNav = useCallback(async (delta: number) => {
    const s = stateRef.current;
    if (s.conversationIndices.length === 0) return;
    const newPos = Math.max(0, Math.min(s.conversationIndices.length - 1, s.cursorPos + delta));
    if (newPos === s.cursorPos) return;
    setCursorPos(newPos);
    setIsEnd(false);
    await loadConversation(s.conversationIndices, newPos);
  }, [loadConversation]);

  const handleToggleTag = useCallback((tag: Tag) => {
    setLocalTags((prev) => {
      const next = prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag];
      stateRef.current.localTags = next;  // keep ref in sync
      return next;
    });
  }, []);

  const handleSourceChange = useCallback((source: string | null) => {
    setSourceFilter(source);
  }, []);

  const handleStatusToggle = useCallback(() => {
    setStatusFilter((prev) => (prev === "unreviewed" ? null : "unreviewed"));
  }, []);

  // Keyboard handlers -- use a stable object that delegates through ref
  const keyHandlers = useRef<Record<string, () => void>>({});
  keyHandlers.current = {
    "1": () => handleRate("bad"),
    "2": () => handleRate("okay"),
    "3": () => handleRate("good"),
    "s": () => handleNav(1),
    "a": () => handleToggleTag("advice-leaking"),
    "l": () => handleToggleTag("too-long"),
    "o": () => handleToggleTag("off-topic"),
    "r": () => handleToggleTag("repetitive"),
    "ArrowLeft": () => handleNav(-1),
    "ArrowRight": () => handleNav(1),
    "u": () => handleStatusToggle(),
  };

  // Stable handler object for useKeyboard
  const [stableHandlers] = useState(() => {
    const h: Record<string, () => void> = {};
    for (const key of ["1", "2", "3", "s", "a", "l", "o", "r", "ArrowLeft", "ArrowRight", "u"]) {
      h[key] = () => keyHandlers.current[key]();
    }
    return h;
  });

  useKeyboard(stableHandlers);

  // Retry handler for connection errors
  const handleRetry = useCallback(() => {
    setConnectionError(false);
    setLoading(true);
    setError(null);
    async function init() {
      try {
        const sourcesResp = await getSources();
        setSources(sourcesResp.sources);
        await fetchList(null, null);
      } catch {
        setConnectionError(true);
        setLoading(false);
      }
    }
    init();
  }, [fetchList]);

  // Connection error state
  if (connectionError) {
    return (
      <div className="center-message">
        <div>Cannot connect to backend at localhost:8000</div>
        <button onClick={handleRetry}>Retry</button>
      </div>
    );
  }

  // Loading state
  if (loading) {
    return (
      <div className="center-message">
        <div>Loading...</div>
      </div>
    );
  }

  // Empty dataset
  if (conversationIndices.length === 0 && !loading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
        <TopBar
          currentPosition={0}
          totalCount={0}
          sourceFilter={sourceFilter}
          sources={sources}
          stats={stats.ratings}
          statusFilter={statusFilter}
          onSourceChange={handleSourceChange}
          onStatusToggle={handleStatusToggle}
        />
        <div className="center-message">
          <div>No conversations to review.</div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <TopBar
        currentPosition={cursorPos + 1}
        totalCount={conversationIndices.length}
        sourceFilter={sourceFilter}
        sources={sources}
        stats={stats.ratings}
        statusFilter={statusFilter}
        onSourceChange={handleSourceChange}
        onStatusToggle={handleStatusToggle}
      />
      {currentConversation ? (
        <ChatView
          messages={currentConversation.messages}
          source={currentConversation.source}
          emotionLabel={currentConversation.emotion_label}
        />
      ) : (
        <div className="chatview center-message">
          <div>Loading conversation...</div>
        </div>
      )}
      <RatingBar
        rating={localRating}
        tags={localTags}
        onRate={handleRate}
        onToggleTag={handleToggleTag}
        error={error}
        isEnd={isEnd}
      />
    </div>
  );
}

export default App;
