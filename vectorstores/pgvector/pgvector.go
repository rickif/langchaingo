package pgvector

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

const (
	// pgLockIDEmbeddingTable is used for advisor lock to fix issue arising from concurrent
	// creation of the embedding table.The same value represents the same lock.
	pgLockIDEmbeddingTable = 1573678846307946494
	// pgLockIDCollectionTable is used for advisor lock to fix issue arising from concurrent
	// creation of the collection table.The same value represents the same lock.
	pgLockIDCollectionTable = 1573678846307946495
	// pgLockIDExtension is used for advisor lock to fix issue arising from concurrent creation
	// of the vector extension. The value is deliberately set to the same as python langchain
	// https://github.com/langchain-ai/langchain/blob/v0.0.340/libs/langchain/langchain/vectorstores/pgvector.py#L167
	pgLockIDExtension = 1573678846307946496
)

var (
	ErrEmbedderWrongNumberVectors = errors.New("number of vectors from embedder does not match number of documents")
	ErrInvalidScoreThreshold      = errors.New("score threshold must be between 0 and 1")
	ErrInvalidFilters             = errors.New("invalid filters")
	ErrUnsupportedOptions         = errors.New("unsupported options")
)

// PGXConn represents both a pgx.Conn and pgxpool.Pool conn.
type PGXConn interface {
	Ping(ctx context.Context) error
	Begin(ctx context.Context) (pgx.Tx, error)
	Exec(ctx context.Context, sql string, arguments ...any) (pgconn.CommandTag, error)
	Query(ctx context.Context, sql string, arguments ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, arguments ...any) pgx.Row
	SendBatch(ctx context.Context, batch *pgx.Batch) pgx.BatchResults
}

type CloseNoErr interface {
	Close()
}

// Store is a wrapper around the pgvector client.
type Store struct {
	embedder            embeddings.Embedder
	connURL             string
	conn                PGXConn
	embeddingTableName  string
	collectionTableName string
	collectionName      string
	collectionUUID      string
	collectionMetadata  map[string]any
	preDeleteCollection bool
	vectorDimensions    int
	hnswIndex           *HNSWIndex
}

type HNSWIndex struct {
	m                int
	efConstruction   int
	distanceFunction string
}

var _ vectorstores.VectorStore = Store{}

// New creates a new Store with options.
func New(ctx context.Context, opts ...Option) (Store, error) {
	store, err := applyClientOptions(opts...)
	if err != nil {
		return Store{}, err
	}
	if store.conn == nil {
		store.conn, err = pgxpool.New(ctx, store.connURL)
		if err != nil {
			return Store{}, err
		}
	}
	if err = store.conn.Ping(ctx); err != nil {
		return Store{}, err
	}
	if err = store.init(ctx); err != nil {
		return Store{}, err
	}
	return store, nil
}

// Close closes the connection.
func (s Store) Close() error {
	if closer, ok := s.conn.(io.Closer); ok {
		return closer.Close()
	}
	if closer, ok := s.conn.(CloseNoErr); ok {
		closer.Close()
	}
	return nil
}

func (s *Store) init(ctx context.Context) error {
	tx, err := s.conn.Begin(ctx)
	if err != nil {
		return err
	}

	if err := s.createVectorExtensionIfNotExists(ctx, tx); err != nil {
		return err
	}
	if err := s.createCollectionTableIfNotExists(ctx, tx); err != nil {
		return err
	}
	if err := s.createEmbeddingTableIfNotExists(ctx, tx); err != nil {
		return err
	}
	if s.preDeleteCollection {
		if err := s.RemoveCollection(ctx, tx); err != nil {
			return err
		}
	}
	if err := s.createOrGetCollection(ctx, tx); err != nil {
		return err
	}

	return tx.Commit(ctx)
}

func (s Store) createVectorExtensionIfNotExists(ctx context.Context, tx pgx.Tx) error {
	// inspired by
	// https://github.com/langchain-ai/langchain/blob/v0.0.340/libs/langchain/langchain/vectorstores/pgvector.py#L167
	// The advisor lock fixes issue arising from concurrent
	// creation of the vector extension.
	// https://github.com/langchain-ai/langchain/issues/12933
	// For more information see:
	// https://www.postgresql.org/docs/16/explicit-locking.html#ADVISORY-LOCKS
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", pgLockIDExtension); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector"); err != nil {
		return err
	}
	return nil
}

func (s Store) createCollectionTableIfNotExists(ctx context.Context, tx pgx.Tx) error {
	// inspired by
	// https://github.com/langchain-ai/langchain/blob/v0.0.340/libs/langchain/langchain/vectorstores/pgvector.py#L167
	// The advisor lock fixes issue arising from concurrent
	// creation of the vector extension.
	// https://github.com/langchain-ai/langchain/issues/12933
	// For more information see:
	// https://www.postgresql.org/docs/16/explicit-locking.html#ADVISORY-LOCKS
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", pgLockIDCollectionTable); err != nil {
		return err
	}
	sql := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
	name varchar,
	cmetadata json,
	"uuid" uuid NOT NULL,
	UNIQUE (name),
	PRIMARY KEY (uuid))`, s.collectionTableName)
	if _, err := tx.Exec(ctx, sql); err != nil {
		return err
	}
	return nil
}

func (s Store) createEmbeddingTableIfNotExists(ctx context.Context, tx pgx.Tx) error {
	// inspired by
	// https://github.com/langchain-ai/langchain/blob/v0.0.340/libs/langchain/langchain/vectorstores/pgvector.py#L167
	// The advisor lock fixes issue arising from concurrent
	// creation of the vector extension.
	// https://github.com/langchain-ai/langchain/issues/12933
	// For more information see:
	// https://www.postgresql.org/docs/16/explicit-locking.html#ADVISORY-LOCKS
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", pgLockIDEmbeddingTable); err != nil {
		return err
	}

	vectorDimensions := ""
	if s.vectorDimensions > 0 {
		vectorDimensions = fmt.Sprintf("(%d)", s.vectorDimensions)
	}

	sql := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
	collection_id uuid,
	embedding vector%s,
	document varchar,
	cmetadata json,
	"uuid" uuid NOT NULL,
	CONSTRAINT langchain_pg_embedding_collection_id_fkey
	FOREIGN KEY (collection_id) REFERENCES %s (uuid) ON DELETE CASCADE,
	PRIMARY KEY (uuid))`, s.embeddingTableName, vectorDimensions, s.collectionTableName)
	if _, err := tx.Exec(ctx, sql); err != nil {
		return err
	}
	sql = fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_collection_id ON %s (collection_id)`, s.embeddingTableName, s.embeddingTableName)
	if _, err := tx.Exec(ctx, sql); err != nil {
		return err
	}

	// See this for more details on HNWS indexes: https://github.com/pgvector/pgvector#hnsw
	if s.hnswIndex != nil {
		sql = fmt.Sprintf(
			`CREATE INDEX IF NOT EXISTS %s_embedding_hnsw ON %s USING hnsw (embedding %s)`,
			s.embeddingTableName, s.embeddingTableName, s.hnswIndex.distanceFunction,
		)
		if s.hnswIndex.m > 0 && s.hnswIndex.efConstruction > 0 {
			sql = fmt.Sprintf("%s WITH (m=%d, ef_construction = %d)", sql, s.hnswIndex.m, s.hnswIndex.efConstruction)
		}
		if _, err := tx.Exec(ctx, sql); err != nil {
			return err
		}
	}

	return nil
}

// AddDocuments adds documents to the Postgres collection associated with 'Store'.
// and returns the ids of the added documents.
func (s Store) AddDocuments(
	ctx context.Context,
	docs []schema.Document,
	options ...vectorstores.Option,
) ([]string, error) {
	opts := s.getOptions(options...)
	if opts.ScoreThreshold != 0 || opts.Filters != nil || opts.NameSpace != "" || opts.DocumentLike != "" {
		return nil, ErrUnsupportedOptions
	}

	docs = s.deduplicate(ctx, opts, docs)

	texts := make([]string, 0, len(docs))
	for _, doc := range docs {
		texts = append(texts, doc.PageContent)
	}

	embedder := s.embedder
	if opts.Embedder != nil {
		embedder = opts.Embedder
	}
	vectors, err := embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return nil, err
	}

	if len(vectors) != len(docs) {
		return nil, ErrEmbedderWrongNumberVectors
	}

	b := &pgx.Batch{}
	sql := fmt.Sprintf(`INSERT INTO %s (uuid, document, embedding, cmetadata, collection_id)
		VALUES($1, $2, $3, $4, $5)`, s.embeddingTableName)

	ids := make([]string, len(docs))
	for docIdx, doc := range docs {
		id := uuid.New().String()
		ids[docIdx] = id
		b.Queue(sql, id, doc.PageContent, pgvector.NewVector(vectors[docIdx]), doc.Metadata, s.collectionUUID)
	}
	return ids, s.conn.SendBatch(ctx, b).Close()
}

//nolint:cyclop
func (s Store) SimilaritySearch(
	ctx context.Context,
	query string,
	numDocuments int,
	options ...vectorstores.Option,
) ([]schema.Document, error) {
	opts := s.getOptions(options...)
	collectionName := s.getNameSpace(opts)
	scoreThreshold, err := s.getScoreThreshold(opts)
	if err != nil {
		return nil, err
	}
	filter, err := s.getFilters(opts)
	if err != nil {
		return nil, err
	}
	embedder := s.embedder
	if opts.Embedder != nil {
		embedder = opts.Embedder
	}
	embedderData, err := embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	whereQuerys := make([]string, 0)
	if scoreThreshold != 0 {
		whereQuerys = append(whereQuerys, fmt.Sprintf("data.distance < %f", 1-scoreThreshold))
	}
	if len(filter) > 0 {
		metaFilter, err := buildMetadataFilter(filter, "data")
		if err != nil {
			return nil, err
		}
		if metaFilter != "" {
			whereQuerys = append(whereQuerys, metaFilter)
		}
	}
	whereQuery := strings.Join(whereQuerys, " AND ")
	if len(whereQuery) == 0 {
		whereQuery = "TRUE"
	}
	dims := len(embedderData)
	sql := fmt.Sprintf(`WITH filtered_embedding_dims AS MATERIALIZED (
    SELECT
        *
    FROM
        %s
    WHERE
        vector_dims (
                embedding
        ) = $1
)
SELECT
	data.document,
	data.cmetadata,
	(1 - data.distance) AS score
FROM (
	SELECT
		filtered_embedding_dims.*,
		embedding <=> $2 AS distance
	FROM
		filtered_embedding_dims
		JOIN %s ON filtered_embedding_dims.collection_id=%s.uuid WHERE %s.name='%s') AS data
WHERE %s
ORDER BY
	data.distance
LIMIT $3`, s.embeddingTableName,
		s.collectionTableName, s.collectionTableName, s.collectionTableName, collectionName,
		whereQuery)
	rows, err := s.conn.Query(ctx, sql, dims, pgvector.NewVector(embedderData), numDocuments)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	docs := make([]schema.Document, 0)
	for rows.Next() {
		doc := schema.Document{}
		if err := rows.Scan(&doc.PageContent, &doc.Metadata, &doc.Score); err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}
	return docs, rows.Err()
}

//nolint:cyclop
func (s Store) Search(
	ctx context.Context,
	limit int,
	offset int,
	options ...vectorstores.Option,
) ([]schema.Document, error) {
	opts := s.getOptions(options...)
	collectionName := s.getNameSpace(opts)
	whereQuery, args, err := s.buildSearchFilters(collectionName, opts)
	if err != nil {
		return nil, err
	}
	orderByQuery, orderByArgs, err := s.buildSearchOrderBy(opts, len(args)+1)
	if err != nil {
		return nil, err
	}
	args = append(args, orderByArgs...)
	sql := fmt.Sprintf(`SELECT
	%s.document,
	%s.cmetadata
FROM %s
JOIN %s ON %s.collection_id=%s.uuid
WHERE %s ORDER BY %s
LIMIT $%d OFFSET $%d`, s.embeddingTableName, s.embeddingTableName, s.embeddingTableName,
		s.collectionTableName, s.embeddingTableName, s.collectionTableName, whereQuery, orderByQuery, len(args)+1, len(args)+2)
	args = append(args, limit, offset)
	rows, err := s.conn.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	docs := make([]schema.Document, 0)
	defer rows.Close()

	for rows.Next() {
		doc := schema.Document{}
		if err := rows.Scan(&doc.PageContent, &doc.Metadata); err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}
	return docs, rows.Err()
}

func (s Store) Count(
	ctx context.Context,
	options ...vectorstores.Option,
) (int, error) {
	opts := s.getOptions(options...)
	collectionName := s.getNameSpace(opts)
	whereQuery, args, err := s.buildSearchFilters(collectionName, opts)
	if err != nil {
		return 0, err
	}

	sql := fmt.Sprintf(`SELECT
	COUNT(1)
FROM %s
JOIN %s ON %s.collection_id=%s.uuid
WHERE %s`, s.embeddingTableName, s.collectionTableName, s.embeddingTableName, s.collectionTableName, whereQuery)

	var count int
	if err := s.conn.QueryRow(ctx, sql, args...).Scan(&count); err != nil {
		return 0, err
	}
	return count, nil
}

func (s Store) DropTables(ctx context.Context) error {
	if _, err := s.conn.Exec(ctx, fmt.Sprintf(`DROP TABLE IF EXISTS %s`, s.embeddingTableName)); err != nil {
		return err
	}
	if _, err := s.conn.Exec(ctx, fmt.Sprintf(`DROP TABLE IF EXISTS %s`, s.collectionTableName)); err != nil {
		return err
	}
	return nil
}

func (s Store) RemoveCollection(ctx context.Context, tx pgx.Tx) error {
	_, err := tx.Exec(ctx, fmt.Sprintf(`DELETE FROM %s WHERE name = $1`, s.collectionTableName), s.collectionName)
	return err
}

func (s *Store) createOrGetCollection(ctx context.Context, tx pgx.Tx) error {
	sql := fmt.Sprintf(`INSERT INTO %s (uuid, name, cmetadata)
		VALUES($1, $2, $3) ON CONFLICT (name) DO
		UPDATE SET cmetadata = $3`, s.collectionTableName)
	if _, err := tx.Exec(ctx, sql, uuid.New().String(), s.collectionName, s.collectionMetadata); err != nil {
		return err
	}
	sql = fmt.Sprintf(`SELECT uuid FROM %s WHERE name = $1 ORDER BY name limit 1`, s.collectionTableName)
	return tx.QueryRow(ctx, sql, s.collectionName).Scan(&s.collectionUUID)
}

// getOptions applies given options to default Options and returns it
// This uses options pattern so clients can easily pass options without changing function signature.
func (s Store) getOptions(options ...vectorstores.Option) vectorstores.Options {
	opts := vectorstores.Options{}
	for _, opt := range options {
		opt(&opts)
	}
	return opts
}

func (s Store) getNameSpace(opts vectorstores.Options) string {
	if opts.NameSpace != "" {
		return opts.NameSpace
	}
	return s.collectionName
}

func (s Store) getScoreThreshold(opts vectorstores.Options) (float32, error) {
	if opts.ScoreThreshold < 0 || opts.ScoreThreshold > 1 {
		return 0, ErrInvalidScoreThreshold
	}
	return opts.ScoreThreshold, nil
}

// getFilters return metadata filters.
// Supports simple key-value pairs (AND-connected) and "$or" key for OR-connected conditions.
// Example:
//
//	map[string]any{
//	    "source": "web",                       // AND condition
//	    "$or": []map[string]any{               // OR group
//	        {"category": "tech"},
//	        {"category": "science"},
//	    },
//	}
func (s Store) getFilters(opts vectorstores.Options) (map[string]any, error) {
	if opts.Filters != nil {
		if filters, ok := opts.Filters.(map[string]any); ok {
			return filters, nil
		}
		return nil, ErrInvalidFilters
	}
	return map[string]any{}, nil
}

// buildMetadataFilter converts a filter map to a SQL WHERE clause fragment using string interpolation.
// Supports "$or" key for OR-connected sub-conditions; all other keys are AND-connected.
func buildMetadataFilter(filter map[string]any, tableAlias string) (string, error) {
	andClauses := make([]string, 0, len(filter))
	for k, v := range filter {
		if k == "$or" {
			orItems, ok := v.([]map[string]any)
			if !ok {
				return "", ErrInvalidFilters
			}
			orClauses := make([]string, 0, len(orItems))
			for _, item := range orItems {
				for ik, iv := range item {
					orClauses = append(orClauses, fmt.Sprintf("(%s.cmetadata ->> '%s') = '%s'", tableAlias, ik, iv))
				}
			}
			if len(orClauses) > 0 {
				andClauses = append(andClauses, "("+strings.Join(orClauses, " OR ")+")")
			}
			continue
		}
		andClauses = append(andClauses, fmt.Sprintf("(%s.cmetadata ->> '%s') = '%s'", tableAlias, k, v))
	}
	return strings.Join(andClauses, " AND "), nil
}

func (s Store) buildSearchFilters(collectionName string, opts vectorstores.Options) (string, []any, error) {
	filter, err := s.getFilters(opts)
	if err != nil {
		return "", nil, err
	}

	whereQuerys := []string{
		fmt.Sprintf("%s.name = $1", s.collectionTableName),
	}
	args := []any{collectionName}

	for k, v := range filter {
		if k == "$or" {
			orItems, ok := v.([]map[string]any)
			if !ok {
				return "", nil, ErrInvalidFilters
			}
			orClauses := make([]string, 0, len(orItems))
			for _, item := range orItems {
				for ik, iv := range item {
					orClauses = append(orClauses,
						fmt.Sprintf("(%s.cmetadata ->> $%d) = $%d", s.embeddingTableName, len(args)+1, len(args)+2),
					)
					args = append(args, ik, iv)
				}
			}
			if len(orClauses) > 0 {
				whereQuerys = append(whereQuerys, "("+strings.Join(orClauses, " OR ")+")")
			}
			continue
		}
		whereQuerys = append(
			whereQuerys,
			fmt.Sprintf("(%s.cmetadata ->> $%d) = $%d", s.embeddingTableName, len(args)+1, len(args)+2),
		)
		args = append(args, k, v)
	}
	if opts.DocumentLike != "" {
		whereQuerys = append(whereQuerys, fmt.Sprintf("%s.document ILIKE $%d", s.embeddingTableName, len(args)+1))
		args = append(args, "%"+opts.DocumentLike+"%")
	}

	return strings.Join(whereQuerys, " AND "), args, nil
}

func (s Store) buildSearchOrderBy(opts vectorstores.Options, argPos int) (string, []any, error) {
	if len(opts.MetadataOrderBy) == 0 {
		return fmt.Sprintf("%s.uuid ASC", s.embeddingTableName), nil, nil
	}

	orderByParts := make([]string, 0, len(opts.MetadataOrderBy)+1)
	orderByArgs := make([]any, 0, len(opts.MetadataOrderBy))
	for idx, key := range opts.MetadataOrderBy {
		orderByParts = append(
			orderByParts,
			fmt.Sprintf("((%s.cmetadata ->> $%d)::bigint) ASC NULLS LAST", s.embeddingTableName, argPos+idx),
		)
		orderByArgs = append(orderByArgs, key)
	}
	orderByParts = append(orderByParts, fmt.Sprintf("%s.uuid ASC", s.embeddingTableName))
	return strings.Join(orderByParts, ", "), orderByArgs, nil
}

func (s Store) deduplicate(
	ctx context.Context,
	opts vectorstores.Options,
	docs []schema.Document,
) []schema.Document {
	if opts.Deduplicater == nil {
		return docs
	}

	filtered := make([]schema.Document, 0, len(docs))
	for _, doc := range docs {
		if !opts.Deduplicater(ctx, doc) {
			filtered = append(filtered, doc)
		}
	}

	return filtered
}

func (s Store) Delete(
	ctx context.Context,
	options ...vectorstores.Option,
) error {
	opts := s.getOptions(options...)
	filter, err := s.getFilters(opts)
	if err != nil {
		return err
	}
	whereQuerys := make([]string, 0)
	for k, v := range filter {
		whereQuerys = append(whereQuerys, fmt.Sprintf("(%s.cmetadata ->> '%s') = '%s'", s.embeddingTableName, k, v))
	}
	whereQuery := strings.Join(whereQuerys, " AND ")
	if len(whereQuery) == 0 {
		whereQuery = "TRUE"
	}

	sql := fmt.Sprintf(`DELETE FROM %s WHERE %s.collection_id= '%s' AND %s`, s.embeddingTableName, s.embeddingTableName, s.collectionUUID, whereQuery)
	_, err = s.conn.Exec(ctx, sql)
	return err
}
