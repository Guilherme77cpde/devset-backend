from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import requests
import json
import os
import uuid
import base64
import mimetypes
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

APP_TITLE = "Devset IA 👾"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# =============================
# MODELOS
# =============================
MODEL_DEFAULT_TEXT = "llama3.2:3b"
MODEL_DEFAULT_VISION = "llava:7b"

MODEL_MAP: Dict[str, str] = {
    "ollama:7b": "llama3.2:3b",
    "llama3.1:8b": "llama3.1:8b",
    "mistral:7b": "mistral:7b",
    "qwen2.5:7b": "qwen2.5:7b",
}

# =============================
# PATHS
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "memory_v2.json")  # ✅ novo
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_HISTORY_MESSAGES = 12
FILES: Dict[str, Dict[str, Any]] = {}

# =============================
# SYSTEM PROMPT
# =============================
SYSTEM_PROMPT = """
Você é a Devset Growth Strategist 👾, especialista em Marketing Digital.

Seu papel:
- Ajudar em estratégias digitais
- Planejamento de crescimento
- Estruturação de produto
- Copy estratégica
- Orientação técnica quando necessário

# REGRAS DE COMPORTAMENTO

- Seja profissional, estruturado e direto.
- Use raciocínio estratégico.
- Seja claro e objetivo.
- Não enrole.
- Se faltar contexto, faça perguntas diretas.

# REGRAS OBRIGATÓRIAS DE FORMATAÇÃO

1) Sempre responder em Markdown limpo.
2) Sempre usar títulos com espaço após # (ex: "## Estratégia").
3) Sempre deixar uma linha em branco após qualquer título.
4) Nunca escrever tudo em bloco único.
5) Parágrafos curtos (máx 3 linhas).
6) Usar listas quando fizer sentido.
7) Nunca juntar título com texto na mesma linha.
8) Se a resposta estiver mal estruturada, reformatar antes de enviar.

# ESTRUTURA PADRÃO (quando aplicável)

## 🎯 Objetivo
(resumo direto do problema)

## 🧠 Estratégia
(explicação organizada)

## 📋 Plano de Ação
1) Passo
2) Passo
3) Passo

## ⚠️ Observações
(pontos importantes)

## 🚀 Próximo Passo
(pergunta estratégica para continuar)

# REGRAS PARA CÓDIGO

- Explique o que será feito antes do código.
- Entregue código completo e funcional.
- Nunca use "...".
- Nunca entregue código parcial.
- Explique como aplicar.

Você é responsável pela clareza da resposta.
Nunca entregue resposta confusa ou desorganizada.
""".strip()

# =============================
# APP
# =============================
app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# FAVICON
# =============================
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# =============================
# HELPERS
# =============================
def resolve_text_model(model_from_ui: Optional[str]) -> str:
    if not model_from_ui:
        return MODEL_DEFAULT_TEXT
    m = model_from_ui.strip()
    if m in MODEL_MAP:
        return MODEL_MAP[m]
    return m

def normalize_text(s: str) -> str:
    return (s or "").replace("\r", "").strip()

def guess_mime(filename: str, content_type: Optional[str]) -> str:
    if content_type:
        return content_type
    mt = mimetypes.guess_type(filename)[0]
    return mt or "application/octet-stream"

def meta_is_image(meta: Dict[str, Any]) -> bool:
    mime = (meta.get("mime") or "").lower()
    return mime.startswith("image/")

def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def sse_data(text: str) -> str:
    """
    SSE exige que cada linha comece com 'data:'.
    Se o chunk tiver '\n', emitir múltiplas linhas 'data:'.
    """
    lines = (text or "").splitlines() or [""]
    return "".join([f"data: {ln}\n" for ln in lines]) + "\n"

def get_ollama_models() -> List[str]:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name")
            if name:
                models.append(name)
        return models
    except:
        return []

# =============================
# MEMORY v2 (perfil + por chat)
# =============================
def memory_default() -> Dict[str, Any]:
    return {
        "version": 2,
        "profiles": {},  # user_id -> {name, goals, niche, prefs...}
        "chats": {},     # chat_id -> {summary, facts, updated_at, turns}
    }

def load_memory_v2() -> Dict[str, Any]:
    if not os.path.exists(MEMORY_FILE):
        return memory_default()
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return memory_default()
            # garante chaves
            if "profiles" not in data or not isinstance(data["profiles"], dict):
                data["profiles"] = {}
            if "chats" not in data or not isinstance(data["chats"], dict):
                data["chats"] = {}
            if "version" not in data:
                data["version"] = 2
            return data
    except:
        return memory_default()

def save_memory_v2(mem: Dict[str, Any]) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)

def ensure_profile(mem: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    prof = mem["profiles"].get(user_id)
    if not isinstance(prof, dict):
        prof = {
            "name": user_id,
            "goals": [],
            "niche": "",
            "tone": "profissional e direto",
            "preferences": [],
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        mem["profiles"][user_id] = prof
    return prof

def ensure_chat(mem: Dict[str, Any], chat_id: str) -> Dict[str, Any]:
    ch = mem["chats"].get(chat_id)
    if not isinstance(ch, dict):
        ch = {
            "summary": "",
            "facts": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "turns": []  # últimos turns brutos (opcional), limitado
        }
        mem["chats"][chat_id] = ch
    return ch

def compact_list(items: List[str], max_items: int = 12, max_len_each: int = 140) -> List[str]:
    out = []
    for it in items or []:
        t = normalize_text(it)
        if not t:
            continue
        if len(t) > max_len_each:
            t = t[:max_len_each].rstrip() + "…"
        out.append(t)
    # remove duplicatas mantendo ordem
    seen = set()
    uniq = []
    for x in out:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq[-max_items:]

# =============================
# “MEMÓRIA INTELIGENTE” (atualiza perfil + chat)
# - usa o próprio Ollama com saída JSON
# =============================
MEMORY_EXTRACTOR_SYSTEM = """
Você é um extrator de memória para um chat.

Tarefa:
1) Atualize o PERFIL do usuário (user_profile) com informações persistentes (nome, objetivo, nicho, preferências de estilo).
2) Atualize a MEMÓRIA DA CONVERSA (chat_memory) com:
   - summary: resumo curto (máx 800 caracteres)
   - facts: lista de fatos úteis (máx 12 itens, cada item máx 140 caracteres)

Regras:
- Retorne APENAS JSON válido.
- Não inclua texto fora do JSON.
- Se não houver nada novo, mantenha os campos atuais.
""".strip()

def ollama_json_call(model: str, messages: List[Dict[str, Any]], timeout: int = 120) -> Dict[str, Any]:
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    content = (j.get("message") or {}).get("content", "")
    content = content.strip()

    # tenta parse direto
    try:
        return json.loads(content)
    except:
        # fallback: tenta achar primeiro/ultimo {}
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end+1])
            except:
                return {}
        return {}

def update_intelligent_memory(
    user_id: str,
    chat_id: str,
    user_text: str,
    assistant_text: str,
    model_for_extractor: str,
) -> None:
    """
    Atualiza memory_v2.json:
    - profiles[user_id]
    - chats[chat_id]
    usando o Ollama pra extrair/resumir.
    """
    mem = load_memory_v2()
    prof = ensure_profile(mem, user_id)
    chat = ensure_chat(mem, chat_id)

    # guarda alguns turns brutos (opcional, limitado)
    chat["turns"] = (chat.get("turns") or [])[-10:]
    chat["turns"].append({"user": user_text, "assistant": assistant_text})

    extractor_messages = [
        {"role": "system", "content": MEMORY_EXTRACTOR_SYSTEM},
        {"role": "user", "content": json.dumps({
            "current_user_profile": prof,
            "current_chat_memory": {
                "summary": chat.get("summary", ""),
                "facts": chat.get("facts", []),
            },
            "new_turn": {
                "user": user_text,
                "assistant": assistant_text
            }
        }, ensure_ascii=False)}
    ]

    data = {}
    try:
        data = ollama_json_call(model_for_extractor, extractor_messages, timeout=120)
    except:
        data = {}

    # aplica atualizações com segurança
    new_prof = data.get("user_profile") if isinstance(data, dict) else None
    new_chat = data.get("chat_memory") if isinstance(data, dict) else None

    if isinstance(new_prof, dict):
        # merge simples
        prof["name"] = normalize_text(new_prof.get("name", prof.get("name", user_id))) or prof.get("name", user_id)

        # goals / preferences listas
        prof["goals"] = compact_list((new_prof.get("goals") or prof.get("goals") or []), max_items=10)
        prof["preferences"] = compact_list((new_prof.get("preferences") or prof.get("preferences") or []), max_items=12)

        niche = normalize_text(new_prof.get("niche", prof.get("niche", "")))
        if niche:
            prof["niche"] = niche

        tone = normalize_text(new_prof.get("tone", prof.get("tone", "")))
        if tone:
            prof["tone"] = tone

        prof["updated_at"] = datetime.now(timezone.utc).isoformat()

    if isinstance(new_chat, dict):
        summary = normalize_text(new_chat.get("summary", chat.get("summary", "")))
        if summary:
            if len(summary) > 800:
                summary = summary[:800].rstrip() + "…"
            chat["summary"] = summary

        facts = new_chat.get("facts", chat.get("facts", []))
        if isinstance(facts, list):
            chat["facts"] = compact_list([str(x) for x in facts], max_items=12, max_len_each=140)

        chat["updated_at"] = datetime.now(timezone.utc).isoformat()

    mem["profiles"][user_id] = prof
    mem["chats"][chat_id] = chat
    save_memory_v2(mem)

def build_memory_context(user_id: str, chat_id: str) -> str:
    """
    Retorna um bloco pequeno pra injetar no prompt:
    - Perfil
    - Resumo + fatos da conversa
    """
    mem = load_memory_v2()
    prof = ensure_profile(mem, user_id)
    chat = ensure_chat(mem, chat_id)

    parts = []
    parts.append("## Memória (Perfil do usuário)")
    parts.append(f"- Nome: {prof.get('name','')}")
    if prof.get("niche"):
        parts.append(f"- Nicho/Contexto: {prof.get('niche')}")
    if prof.get("tone"):
        parts.append(f"- Tom preferido: {prof.get('tone')}")
    goals = prof.get("goals") or []
    if goals:
        parts.append("- Objetivos:")
        for g in goals[:8]:
            parts.append(f"  - {g}")
    prefs = prof.get("preferences") or []
    if prefs:
        parts.append("- Preferências:")
        for p in prefs[:8]:
            parts.append(f"  - {p}")

    parts.append("\n## Memória (Conversa atual)")
    if chat.get("summary"):
        parts.append(f"- Resumo: {chat.get('summary')}")
    facts = chat.get("facts") or []
    if facts:
        parts.append("- Fatos úteis:")
        for f in facts[:12]:
            parts.append(f"  - {f}")

    return "\n".join(parts).strip()

# =============================
# BUILD MESSAGES (com memória inteligente)
# =============================
def build_messages(
    user_text: str,
    user_id: str,
    chat_id: str,
    use_intelligent_memory: bool,
    history: Optional[List[Dict[str, Any]]],
    file_ids: Optional[List[str]]
) -> Tuple[List[Dict[str, Any]], bool]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ✅ injeta memória compacta no início (se ativado)
    if use_intelligent_memory:
        mem_context = build_memory_context(user_id, chat_id)
        if mem_context:
            messages.append({"role": "system", "content": mem_context})

    # histórico vindo do front (últimas mensagens visíveis)
    if history:
        for h in history[-MAX_HISTORY_MESSAGES:]:
            if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str):
                messages.append({"role": h["role"], "content": h["content"]})

    # arquivos/imagens
    file_ids = file_ids or []
    images_b64: List[str] = []
    for fid in file_ids:
        meta = FILES.get(fid)
        if not meta:
            continue
        if meta_is_image(meta):
            images_b64.append(image_to_b64(meta["path"]))

    if images_b64:
        messages.append({"role": "user", "content": user_text, "images": images_b64})
        return messages, True

    messages.append({"role": "user", "content": user_text})
    return messages, False

# =============================
# OLLAMA STREAM (SSE)
# =============================
def stream_ollama(messages: List[Dict[str, Any]], model_name: str):
    payload = {"model": model_name, "messages": messages, "stream": True}
    try:
        with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=240) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("done"):
                    break
                chunk = (obj.get("message") or {}).get("content", "")
                if chunk:
                    yield sse_data(chunk)
    except Exception as e:
        yield sse_data(f"Erro 👾: {str(e)}")

    yield "event: done\ndata: [DONE]\n\n"

# =============================
# SCHEMAS
# =============================
class ChatInput(BaseModel):
    message: str
    # ✅ novos
    user_id: Optional[str] = "guilherme"
    chat_id: Optional[str] = None

    # ✅ liga/desliga memória inteligente
    smart_memory: Optional[bool] = True

    file_ids: Optional[List[str]] = None
    history: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None

# =============================
# ROUTES
# =============================
@app.get("/")
def home():
    real_models = get_ollama_models()
    models_ui = real_models if real_models else list(MODEL_MAP.keys())
    return {
        "status": "Devset API online 🚀",
        "model_default_text": MODEL_DEFAULT_TEXT,
        "model_default_vision": MODEL_DEFAULT_VISION,
        "models_available_ui": models_ui,
        "stream": "/chat_stream",
        "upload": "/upload",
        "memory": "/memory",
        "memory_chat": "/memory/chat/{chat_id}",
    }

@app.get("/memory")
def memory_overview():
    mem = load_memory_v2()
    return {
        "version": mem.get("version", 2),
        "profiles_count": len(mem.get("profiles", {})),
        "chats_count": len(mem.get("chats", {})),
    }

@app.get("/memory/chat/{chat_id}")
def memory_chat(chat_id: str):
    mem = load_memory_v2()
    ch = mem.get("chats", {}).get(chat_id, {})
    return ch or {}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    raw = await file.read()

    path = os.path.join(UPLOAD_DIR, file_id)
    with open(path, "wb") as f:
        f.write(raw)

    mime = guess_mime(file.filename, file.content_type)
    FILES[file_id] = {"path": path, "name": file.filename, "mime": mime}

    return {"file_id": file_id}

@app.post("/chat_stream")
def chat_stream(payload: ChatInput):
    user_text = (payload.message or "").strip()
    if not user_text:
        def gen_empty():
            yield sse_data("Mensagem vazia.")
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(gen_empty(), media_type="text/event-stream")

    user_id = (payload.user_id or "guilherme").strip() or "guilherme"
    chat_id = (payload.chat_id or "").strip() or f"chat_{uuid.uuid4().hex[:10]}"

    chosen_model = resolve_text_model(payload.model)
    messages, use_vision = build_messages(
        user_text=user_text,
        user_id=user_id,
        chat_id=chat_id,
        use_intelligent_memory=bool(payload.smart_memory),
        history=payload.history,
        file_ids=payload.file_ids
    )

    model_to_use = MODEL_DEFAULT_VISION if use_vision else chosen_model

    full_response: List[str] = []

    def generator():
        # manda um “meta” inicial opcional (não obrigatório pro front)
        yield sse_data(f"[meta] chat_id={chat_id}")

        for sse in stream_ollama(messages, model_to_use):
            if sse.startswith("data: "):
                full_response.append(sse[6:])  # NÃO strip
            yield sse

        # ✅ atualiza memória inteligente no final
        assistant_text = normalize_text("".join(full_response))
        if payload.smart_memory:
            try:
                # usa o mesmo modelo de texto pra extrair memória
                update_intelligent_memory(
                    user_id=user_id,
                    chat_id=chat_id,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    model_for_extractor=chosen_model or MODEL_DEFAULT_TEXT
                )
            except:
                pass

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(generator(), media_type="text/event-stream", headers=headers)
