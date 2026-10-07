# Splitting documents into chunks

Chunking divides a document into smaller pieces that a retriever can rank.
ragstat will start with fixed windows of whitespace-delimited words.

Chunk size is the maximum number of words in one chunk. Overlap is the number
of words shared by neighboring windows. For a chunk size of five and an
overlap of two, the next window starts three words after the previous one.

Overlap must be nonnegative and smaller than the chunk size. It helps keep
context near a boundary available to both neighboring chunks. The final window
may contain fewer words than the configured chunk size.

Every chunk keeps its parent document ID so that evaluation can count a
document once even when several of its chunks are retrieved.
