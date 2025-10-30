import psycopg2
from psycopg2.extras import execute_batch


class DBHelper:
    def __init__(self, dsn):
        self.dsn = dsn
        self.conn = psycopg2.connect(dsn)
        self.ensure_tables()
        self.ensure_indexes()

    def ensure_indexes(self):
        # Create HNSW indexes on user_embeddings and post_embeddings if not exist
        with self.conn.cursor() as cur:
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename='user_embeddings' AND indexname='idx_user_embedding_hnsw') THEN
                        CREATE INDEX idx_user_embedding_hnsw ON user_embeddings USING hnsw (embedding vector_cosine_ops);
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename='post_embeddings' AND indexname='idx_post_embedding_hnsw') THEN
                        CREATE INDEX idx_post_embedding_hnsw ON post_embeddings USING hnsw (embedding vector_cosine_ops);
                    END IF;
                END
                $$;
            """)
            self.conn.commit()

    def ensure_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_embeddings (
                    id SERIAL PRIMARY KEY,
                    user_id INT UNIQUE NOT NULL,
                    embedding vector(64) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS post_embeddings (
                    id SERIAL PRIMARY KEY,
                    post_id INT UNIQUE NOT NULL,
                    embedding vector(64) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users_raw (
                    user_id INT PRIMARY KEY,
                    age DOUBLE PRECISION,
                    gender INT,
                    num_friends INT,
                    avg_likes_received DOUBLE PRECISION,
                    avg_comments_received DOUBLE PRECISION,
                    avg_shares_received DOUBLE PRECISION,
                    active_days_last_week INT,
                    time_spent_last_week DOUBLE PRECISION,
                    num_groups INT,
                    has_profile_picture DOUBLE PRECISION
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS posts_raw (
                    post_id INT PRIMARY KEY,
                    post_length DOUBLE PRECISION,
                    num_images INT,
                    num_videos INT,
                    num_hashtags INT,
                    author_followers INT,
                    author_following INT,
                    author_posts_last_week INT,
                    is_boosted INT,
                    post_type varchar(8),
                    post_time_hour DOUBLE PRECISION
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS interactions_raw (
                    user_id INT,
                    post_id INT,
                    label INT,
                    PRIMARY KEY (user_id, post_id)
                );
            """)
            self.conn.commit()

    def insert_user_embedding(self, user_id, embedding):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_embeddings (user_id, embedding) VALUES (%s, %s)",
                (user_id, embedding)
            )
            self.conn.commit()

    def insert_post_embedding(self, post_id, embedding):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO post_embeddings (post_id, embedding) VALUES (%s, %s)",
                (post_id, embedding)
            )
            self.conn.commit()

    def insert_user_embeddings_batch(self, records):
        # records: list of (user_id, embedding)
        with self.conn.cursor() as cur:
            execute_batch(
                cur,
                "INSERT INTO user_embeddings (user_id, embedding) VALUES (%s, %s)",
                records
            )
            self.conn.commit()

    def insert_post_embeddings_batch(self, records):
        # records: list of (post_id, embedding)
        with self.conn.cursor() as cur:
            execute_batch(
                cur,
                "INSERT INTO post_embeddings (post_id, embedding) VALUES (%s, %s)",
                records
            )
            self.conn.commit()

    def query_similar_users(self, embedding, top_k=5):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, embedding <=> %s AS distance
                FROM user_embeddings
                ORDER BY distance
                LIMIT %s
                """,
                (embedding, top_k)
            )
            return cur.fetchall()

    def query_similar_posts(self, embedding, top_k=5):
        with self.conn.cursor() as cur:
            # pgvector requires explicit cast from Python list/array to vector type
            cur.execute(
                """
                SELECT post_id, embedding <=> %s::vector AS distance
                FROM post_embeddings
                ORDER BY distance
                LIMIT %s
                """,
                (embedding, top_k)
            )
            return cur.fetchall()

    def clear_user_embeddings(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM user_embeddings;")
            self.conn.commit()

    def clear_post_embeddings(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM post_embeddings;")
            self.conn.commit()

    def clear_post_raw(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM posts_raw;")
            self.conn.commit()

    def clear_user_raw(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM users_raw;")
            self.conn.commit()
   
    def clear_interactions_raw(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM interactions_raw;")
            self.conn.commit()    

    def close(self):
        self.conn.close()

    def query_similar_users_ann(self, embedding, top_k=5):
        # Approximate nearest neighbor search using HNSW index
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, embedding <=> %s::vector AS distance
                FROM user_embeddings
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k)
            )
            return cur.fetchall()

    def query_similar_posts_ann(self, embedding, top_k=5):
        # Approximate nearest neighbor search using HNSW index
        
        with self.conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT post_id, embedding <=> %s::vector AS distance
                    FROM post_embeddings
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                    """,
                    (embedding, embedding, top_k)
                )
            except psycopg2.Error as e:
                # 1. Rollback the transaction to reset the connection state
                self.conn.rollback() 
                
                # 2. Log the error (optional but recommended)
                print(f"Database error in query_similar_users_ann: {e}")
                
                # 3. Re-raise a standard exception so FastAPI can handle it
                raise ConnectionError("Database query failed after rollback.")
            return cur.fetchall()

    def insert_dataframe(self, table_name: str, df):
        """
        Generic method to insert a pandas DataFrame into a table.
        """
        cols = list(df.columns)
        placeholders = ', '.join(['%s'] * len(cols))
        sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({placeholders})"
        records = [tuple(row) for row in df.to_numpy()]
        with self.conn.cursor() as cur:
            execute_batch(cur, sql, records)
            self.conn.commit()