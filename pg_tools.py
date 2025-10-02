import os
from dotenv import load_dotenv
import psycopg2
from typing import Optional
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  

def get_conn():
    return psycopg2.connect(DATABASE_URL)

# Essa classe garante que o objeto de Python passe todos esses campos
class AddTransactionArgs(BaseModel):
    amount: float = Field(..., description="Valor da transação (use positivo).")
    source_text: str = Field(..., description="Texto original do usuário.")
    occurred_at: Optional[str] = Field(
        default=None,
        description="Timestamp ISO 8601; se ausente, usa NOW() no banco."
    )
    type_id: Optional[int] = Field(default=None, description="ID em transaction_types (1=INCOME, 2=EXPENSES, 3=TRANSFER).")
    type_name: Optional[str] = Field(default=None, description="Nome do tipo em inglês: INCOME | EXPENSES | TRANSFER.")
    category_id: Optional[int] = Field(default=None, description="FK de categories (opcional).")
    description: Optional[str] = Field(default=None, description="Descrição (opcional).")
    payment_method: Optional[str] = Field(default=None, description="Forma de pagamento (opcional).")

TYPE_ALIASES = {"INCOME": "INCOME", "ENTRADA": "INCOME", "GANHEI": "INCOME", "RECEITA": "INCOME", "SALÁRIO": "INCOME", "EXPENSE": "EXPENSES", "EXPENSES": "EXPENSES", "GASTO": "EXPENSES", "COMPREI": "EXPENSES", "COMPRA": "EXPENSES", "GASTEI": "EXPENSES", "TRANSFERENCIA": "TRANSFER", "TRANSFER": "TRANSFER"}

#Garante que o campo type da tabela transactions receba um id válido (1=INCOME, 2=EXPENSES, 3=TRANSFER
def _resolve_type_id(cur, type_id: Optional[int], type_name: Optional[str]) -> Optional[int]:
    if type_name:
        t = type_name.strip().upper()
        if t in TYPE_ALIASES:
            t = TYPE_ALIASES[t]
        cur.execute("SELECT id FROM transaction_types WHERE UPPER(type)=%s LIMIT 1;", (t,))
        row = cur.fetchone()
        return row[0] if row else None
    if type_id:
        return int(type_id)
    return 2


# Tool: add_transaction
@tool("add_transaction", args_schema=AddTransactionArgs)
def add_transaction(
    amount: float,
    source_text: str,
    occurred_at: Optional[str] = None,
    type_id: Optional[int] = None,
    type_name: Optional[str] = None,
    category_id: Optional[int] = None,
    description: Optional[str] = None,
    payment_method: Optional[str] = None,
) -> dict:
    """
    Insere uma transação financeira no banco de dados Postgres.
    """ # docstring obrigatório da @tools do langchain (estranho, mas legal né?)
    conn = get_conn()
    cur = conn.cursor()
    try:
        resolved_type_id = _resolve_type_id(cur, type_id, type_name)
        if not resolved_type_id:
            return {"status": "error", "message": "Tipo inválido (use type_id ou type_name: INCOME/EXPENSES/TRANSFER)."}

        if occurred_at:
            cur.execute(
                """
                INSERT INTO transactions
                    (amount, type, category_id, description, payment_method, occurred_at, source_text)
                VALUES
                    (%s, %s, %s, %s, %s, %s::timestamptz, %s)
                RETURNING id, occurred_at;
                """,
                (amount, resolved_type_id, category_id, description, payment_method, occurred_at, source_text),
            )
        else:
            cur.execute(
                """
                INSERT INTO transactions
                    (amount, type, category_id, description, payment_method, occurred_at, source_text)
                VALUES
                    (%s, %s, %s, %s, %s, NOW(), %s)
                RETURNING id, occurred_at;
                """,
                (amount, resolved_type_id, category_id, description, payment_method, source_text),
            )

        new_id, occurred = cur.fetchone()
        conn.commit()
        return {"status": "ok", "id": new_id, "occurred_at": str(occurred)}

    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


class QueryTransactionsArgs(BaseModel):
    text: Optional[str] = Field(None, description="Filtro por texto (description/source_text).")
    type_name: Optional[str] = Field(None, description="Nome do tipo: INCOME | EXPENSES | TRANSFER.")
    date_local: Optional[str] = Field(None, description="Data específica (YYYY-MM-DD).")
    date_from_local: Optional[str] = Field(None, description="Data inicial (YYYY-MM-DD).")
    date_to_local: Optional[str] = Field(None, description="Data final (YYYY-MM-DD).")
    limit: int = Field(20, description="Número máximo de transações para retornar.")

@tool("query_transactions", args_schema=QueryTransactionsArgs)
def query_transactions(
    text: Optional[str] = None,
    type_name: Optional[str] = None,
    date_local: Optional[str] = None,
    date_from_local: Optional[str] = None,
    date_to_local: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """
    Consulta transações com filtros opcionais.
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        query = """
            SELECT id, amount, type, category_id, description, payment_method, occurred_at, source_text
            FROM transactions
            WHERE 1=1
        """
        params = []

        if text:
            query += " AND (description ILIKE %s OR source_text ILIKE %s)"
            params.extend([f"%{text}%", f"%{text}%"])
        if type_name:
            query += " AND type = (SELECT id FROM transaction_types WHERE UPPER(type)=%s)"
            params.append(type_name.upper())
        if date_local:
            query += " AND DATE(occurred_at) = %s"
            params.append(date_local)
        if date_from_local and date_to_local:
            query += " AND DATE(occurred_at) BETWEEN %s AND %s"
            params.extend([date_from_local, date_to_local])

        query += " ORDER BY occurred_at DESC LIMIT %s"
        params.append(limit)

        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        return {"transactions": rows}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool("total_balance")
def total_balance() -> dict:
    """
    Retorna o saldo total.
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                COALESCE(SUM(
                    CASE 
                        WHEN t.type = 1 THEN amount
                        WHEN t.type = 2 THEN -amount
                        ELSE 0
                    END
                ), 0) as balance
            FROM transactions t
        """)
        balance = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"total_balance": balance}
    except Exception as e:
        return {"status": "error", "message": str(e)}


class DailyBalanceArgs(BaseModel):
    date_local: str = Field(..., description="Data (YYYY-MM-DD).")

@tool("daily_balance", args_schema=DailyBalanceArgs)
def daily_balance(date_local: str) -> dict:
    """
    Retorna o saldo de um dia específico.
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                COALESCE(SUM(
                    CASE 
                        WHEN t.type = 1 THEN amount
                        WHEN t.type = 2 THEN -amount
                        ELSE 0
                    END
                ), 0) as balance
            FROM transactions t
            WHERE DATE(occurred_at) = %s
        """, (date_local,))
        balance = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"daily_balance": balance}
    except Exception as e:
        return {"status": "error", "message": str(e)}


TOOLS = [add_transaction, query_transactions, total_balance, daily_balance]