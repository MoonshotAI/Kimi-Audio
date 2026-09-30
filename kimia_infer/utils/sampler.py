import torch


def _apply_repetition_penalty(logits, recent_tokens, penalty, window_size):
    """Penalize each recent token once, independently for each batch row.

    A history of shape [history_len] or [1, history_len] is shared by all
    rows; [batch_size, history_len] supplies a separate history per row.
    A non-positive window disables the penalty. The input logits are not
    modified, including when they are a view of a sequence's last step.
    """
    if penalty <= 1.0 or recent_tokens is None or window_size <= 0:
        return logits
    if recent_tokens.ndim == 1:
        recent_tokens = recent_tokens.unsqueeze(0)
    if recent_tokens.ndim != 2 or recent_tokens.size(0) not in (1, logits.size(0)):
        raise ValueError(
            "recent_tokens must have shape [history_len], [1, history_len], "
            "or [batch_size, history_len]"
        )
    if recent_tokens.size(1) == 0:
        return logits

    recent_window = recent_tokens[:, -window_size:].to(
        device=logits.device, dtype=torch.long
    ).expand(logits.size(0), -1)
    scores = logits.gather(1, recent_window)
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    # Duplicate token IDs gather the same original score, so their penalty
    # does not compound. Out-of-place scatter also preserves caller logits.
    return logits.scatter(1, recent_window, scores)


class KimiASampler:
    def __init__(
        self,
        audio_top_k: int,
        audio_temperature: float,
        audio_repetition_penalty: float,
        audio_repetition_window_size: int,
        text_top_k: int,
        text_temperature: float,
        text_repetition_penalty: float,
        text_repetition_window_size: int,
    ):
        self.audio_top_k = audio_top_k
        self.audio_temperature = audio_temperature
        self.text_top_k = text_top_k
        self.text_temperature = text_temperature

        self.audio_repetition_penalty = audio_repetition_penalty
        self.audio_repetition_window_size = audio_repetition_window_size
        self.text_repetition_penalty = text_repetition_penalty
        self.text_repetition_window_size = text_repetition_window_size

    def sample_audio_logits(
        self, logits: torch.Tensor, recent_tokens=None
    ) -> torch.Tensor:
        """Sample from audio logits with top-k, temperature and repetition penalty.

        Args:
            logits: Logits tensor of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            recent_tokens: Optional shared history [history_len] or [1, history_len],
                or a per-row history [batch_size, history_len]. The last window_size
                tokens are penalized; shorter histories are used in full.

        Returns:
            Sampled token ids
        """
        # Take the last token's logits if we have a sequence dimension
        if len(logits.shape) == 3:
            logits = logits[:, -1]

        logits = _apply_repetition_penalty(
            logits,
            recent_tokens,
            self.audio_repetition_penalty,
            self.audio_repetition_window_size,
        )

        # Convert to probabilities with softmax
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Apply temperature scaling if not greedy
        if self.audio_temperature > 1e-6:
            logprobs = logprobs / self.audio_temperature

            # Apply top-k sampling
            if self.audio_top_k > 0:
                # Get probabilities from logprobs
                probs = torch.exp(logprobs)

                # Select top-k probabilities and indices
                top_k_probs, top_k_indices = torch.topk(probs, self.audio_top_k, dim=-1)

                # Sample from the top-k distribution
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(
                    1
                )
                next_token = top_k_indices.gather(
                    -1, sampled_indices.unsqueeze(-1)
                ).squeeze(-1)
            else:
                # Sample from the full distribution
                next_token = torch.multinomial(
                    torch.exp(logprobs), num_samples=1
                ).squeeze(1)
        else:
            # Greedy sampling (temperature = 0)
            next_token = torch.argmax(logprobs, dim=-1)

        return next_token

    def sample_text_logits(
        self, logits: torch.Tensor, recent_tokens=None
    ) -> torch.Tensor:
        """Sample from text logits with top-k, temperature and repetition penalty.

        Args:
            logits: Logits tensor of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            recent_tokens: Optional shared history [history_len] or [1, history_len],
                or a per-row history [batch_size, history_len]. The last window_size
                tokens are penalized; shorter histories are used in full.

        Returns:
            Sampled token ids
        """
        # Take the last token's logits if we have a sequence dimension
        if len(logits.shape) == 3:
            logits = logits[:, -1]

        logits = _apply_repetition_penalty(
            logits,
            recent_tokens,
            self.text_repetition_penalty,
            self.text_repetition_window_size,
        )

        # Convert to probabilities with softmax
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Apply temperature scaling if not greedy
        if self.text_temperature > 1e-6:
            logprobs = logprobs / self.text_temperature

            # Apply top-k sampling
            if self.text_top_k > 0:
                # Get probabilities from logprobs
                probs = torch.exp(logprobs)

                # Select top-k probabilities and indices
                top_k_probs, top_k_indices = torch.topk(probs, self.text_top_k, dim=-1)

                # Sample from the top-k distribution
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(
                    1
                )
                next_token = top_k_indices.gather(
                    -1, sampled_indices.unsqueeze(-1)
                ).squeeze(-1)
            else:
                # Sample from the full distribution
                next_token = torch.multinomial(
                    torch.exp(logprobs), num_samples=1
                ).squeeze(1)
        else:
            # Greedy sampling (temperature = 0)
            next_token = torch.argmax(logprobs, dim=-1)

        return next_token
