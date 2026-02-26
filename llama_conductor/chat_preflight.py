from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


def handle_preflight(
    *,
    state: Any,
    session_id: str,
    stream: bool,
    selector: str,
    user_text: str,
    user_text_raw: str,
    raw_messages: List[Dict[str, Any]],
    split_selector: Callable[[str], Tuple[str, str]],
    is_command: Callable[[str], bool],
    requires_sensitive_confirm: Callable[[Any, str], bool],
    handle_command: Callable[..., Optional[str]],
    maybe_capture_command_output: Optional[Callable[..., None]],
    soft_alias_command: Callable[[str, Any], Optional[str]],
    make_stream_response: Callable[[str], Any],
    make_json_response: Callable[[str], Any],
) -> Tuple[bool, Optional[Any], str, str, str, List[Dict[str, Any]], bool]:
    """Pre-pipeline gate handling.

    Returns:
      (handled, response, selector, user_text, user_text_raw, raw_messages, sensitive_override_once)
    """
    sensitive_override_once = False

    if state.pending_sensitive_confirm_query:
        yn = (user_text_raw or "").strip().lower()
        if yn in ("y", "yes"):
            resumed_query = state.pending_sensitive_confirm_query
            state.pending_sensitive_confirm_query = ""
            user_text_raw = resumed_query
            selector, user_text = split_selector(resumed_query)
            sensitive_override_once = True
            for i in range(len(raw_messages) - 1, -1, -1):
                if raw_messages[i].get("role") == "user":
                    raw_messages[i]["content"] = resumed_query
                    break
        elif yn in ("n", "no"):
            state.pending_sensitive_confirm_query = ""
            text = "[router] Sensitive request cancelled"
            return True, make_stream_response(text) if stream else make_json_response(text), selector, user_text, user_text_raw, raw_messages, sensitive_override_once
        else:
            state.pending_sensitive_confirm_query = ""
            text = "[router] Sensitive request cancelled (default N)"
            return True, make_stream_response(text) if stream else make_json_response(text), selector, user_text, user_text_raw, raw_messages, sensitive_override_once

    if selector == "":
        try:
            cmd_reply = handle_command(user_text_raw, state=state, session_id=session_id)
        except Exception as e:
            text = f"[router error: command handler crashed: {e.__class__.__name__}: {e}]"
            return True, make_stream_response(text) if stream else make_json_response(text), selector, user_text, user_text_raw, raw_messages, sensitive_override_once

        if cmd_reply is not None:
            text = cmd_reply
            if maybe_capture_command_output is not None:
                try:
                    maybe_capture_command_output(
                        session_id=session_id,
                        state=state,
                        cmd_text=user_text_raw,
                        reply_text=text,
                    )
                except Exception:
                    pass
            return True, make_stream_response(text) if stream else make_json_response(text), selector, user_text, user_text_raw, raw_messages, sensitive_override_once

        alias_cmd = soft_alias_command(user_text_raw, state)
        if alias_cmd:
            try:
                alias_reply = handle_command(alias_cmd, state=state, session_id=session_id)
            except Exception as e:
                text = f"[router error: soft alias crashed: {e.__class__.__name__}: {e}]"
                return True, make_stream_response(text) if stream else make_json_response(text), selector, user_text, user_text_raw, raw_messages, sensitive_override_once
            if alias_reply is not None:
                text = alias_reply
                if maybe_capture_command_output is not None:
                    try:
                        maybe_capture_command_output(
                            session_id=session_id,
                            state=state,
                            cmd_text=alias_cmd,
                            reply_text=text,
                        )
                    except Exception:
                        pass
                return True, make_stream_response(text) if stream else make_json_response(text), selector, user_text, user_text_raw, raw_messages, sensitive_override_once

    if (
        not sensitive_override_once
        and not is_command(user_text_raw)
        and requires_sensitive_confirm(state, user_text_raw)
    ):
        state.pending_sensitive_confirm_query = user_text_raw
        text = "[router] This request is sensitive in a professional context. Continue anyway? [Y/N]"
        return True, make_stream_response(text) if stream else make_json_response(text), selector, user_text, user_text_raw, raw_messages, sensitive_override_once

    return False, None, selector, user_text, user_text_raw, raw_messages, sensitive_override_once
