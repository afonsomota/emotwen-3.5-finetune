export type Rating = "good" | "okay" | "bad";
export type Tag = "advice-leaking" | "too-long" | "off-topic" | "repetitive";

export const ALL_TAGS: Tag[] = ["advice-leaking", "too-long", "off-topic", "repetitive"];

export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ConversationListItem {
  idx: number;
  source: string;
  emotion_label: string;
  n_turns: number;
  preview: string;
  rating: Rating | null;
  tags: Tag[];
}

export interface Conversation {
  idx: number;
  source: string;
  emotion_label: string;
  messages: Message[];
  feedback: Feedback | null;
}

export interface Feedback {
  conversation_idx?: number;
  rating: Rating;
  tags: Tag[];
  created_at: string;
  updated_at: string;
}

export interface Stats {
  total: number;
  reviewed: number;
  ratings: Record<Rating, number>;
  tags: Record<Tag, number>;
  by_source?: Record<string, { total: number; reviewed: number }>;
}

export interface ConversationListResponse {
  total: number;
  offset: number;
  limit: number;
  items: ConversationListItem[];
}
