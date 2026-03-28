import type { ConversationListResponse, Conversation, Feedback, Stats, Rating, Tag } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // ignore parse errors
    }
    if (res.status === 503) {
      throw new Error("loading: " + detail);
    }
    throw new Error(detail);
  }
  return res.json();
}

export function getSources(): Promise<{ sources: string[] }> {
  return request("/api/sources");
}

export function getConversations(params: {
  source?: string | null;
  status?: string | null;
  offset?: number;
  limit?: number;
}): Promise<ConversationListResponse> {
  const searchParams = new URLSearchParams();
  if (params.source) searchParams.set("source", params.source);
  if (params.status) searchParams.set("status", params.status);
  if (params.offset !== undefined) searchParams.set("offset", String(params.offset));
  if (params.limit !== undefined) searchParams.set("limit", String(params.limit));
  const qs = searchParams.toString();
  return request(`/api/conversations${qs ? "?" + qs : ""}`);
}

export function getConversation(idx: number): Promise<Conversation> {
  return request(`/api/conversations/${idx}`);
}

export function postFeedback(data: {
  conversation_idx: number;
  rating: Rating;
  tags: Tag[];
}): Promise<Feedback> {
  return request("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export function getStats(source?: string | null): Promise<Stats> {
  const qs = source ? `?source=${encodeURIComponent(source)}` : "";
  return request(`/api/stats${qs}`);
}
