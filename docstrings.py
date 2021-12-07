
_doc_param_input = \
    """Path to VCF file on the local file system. May be uncompressed or gzip-compatible
        compressed file. May also be a file-like object (e.g., `io.BytesIO`)."""

_doc_param_fields = \
    """Fields to extract data for. Should be a list of strings, e.g., ``['variants/CHROM',
        'variants/POS', 'variants/DP', 'calldata/GT']``. If you are feeling lazy,
        you can drop the 'variants/' and 'calldata/' prefixes, in which case the fields
        will be matched against fields declared in the VCF header, with variants taking
        priority over calldata if a field with the same ID exists both in INFO and
        FORMAT headers. I.e., ``['CHROM', 'POS', 'DP', 'GT']`` will work, although
        watch out for fields like 'DP' which can be both INFO and FORMAT. For
        convenience, some special string values are also recognized. To extract all
        fields, provide just the string ``'*'``. To extract all variants fields
        (including all INFO fields) provide ``'variants/*'``. To extract all calldata
        fields (i.e., defined in FORMAT headers) provide ``'calldata/*'``."""

_doc_param_exclude_fields = \
    """Fields to exclude. E.g., for use in combination with ``fields='*'``."""

_doc_param_rename_fields = \
    """Fields to be renamed. Should be a dictionary mapping old to new names,
    giving the complete path, e.g., ``{'variants/FOO': 'variants/bar'}``."""

_doc_param_types = \
    """Overide data types. Should be a dictionary mapping field names to NumPy data types.
        E.g., providing the dictionary ``{'variants/DP': 'i8', 'calldata/GQ': 'i2'}`` will
        mean the 'variants/DP' field is stored in a 64-bit integer array, and the
        'calldata/GQ' field is stored in a 16-bit integer array."""

_doc_param_numbers = \
    """Override the expected number of values. Should be a dictionary mapping field names
        to integers. E.g., providing the dictionary ``{'variants/ALT': 5,
        'variants/AC': 5, 'calldata/HQ': 2}`` will mean that, for each variant, 5 values
        are stored for the 'variants/ALT' field, 5 values are stored for the
        'variants/AC' field, and for each sample, 2 values are stored for the
        'calldata/HQ' field."""

_doc_param_alt_number = \
    """Assume this number of alternate alleles and set expected number of values
        accordingly for any field declared with number 'A' or 'R' in the VCF
        meta-information."""

_doc_param_fills = \
    """Override the fill value used for empty values. Should be a dictionary mapping
        field names to fill values."""

_doc_param_region = \
    """Genomic region to extract variants for. If provided, should be a tabix-style
        region string, which can be either just a chromosome name (e.g., '2L'),
        or a chromosome name followed by 1-based beginning and end coordinates (e.g.,
        '2L:100000-200000'). Note that only variants whose start position (POS) is
        within the requested range will be included. This is slightly different from
        the default tabix behaviour, where a variant (e.g., deletion) may be included
        if its position (POS) occurs before the requested region but its reference allele
        overlaps the region - such a variant will *not* be included in the data
        returned by this function."""

_doc_param_tabix = \
    """Name or path to tabix executable. Only required if `region` is given. Setting
        `tabix` to `None` will cause a fall-back to scanning through the VCF file from
        the beginning, which may be much slower than tabix but the only option if tabix
        is not available on your system and/or the VCF file has not been tabix-indexed."""

_doc_param_samples = \
    """Selection of samples to extract calldata for. If provided, should be a list of
        strings giving sample identifiers. May also be a list of integers giving
        indices of selected samples."""

_doc_param_transformers = \
    """Transformers for post-processing data. If provided, should be a list of Transformer
        objects, each of which must implement a "transform()" method that accepts a dict
        containing the chunk of data to be transformed. See also the
        :class:`ANNTransformer` class which implements post-processing of data from
        SNPEFF."""

_doc_param_buffer_size = \
    """Size in bytes of the I/O buffer used when reading data from the underlying file or
        tabix stream."""

_doc_param_chunk_length = \
    """Length (number of variants) of chunks in which data are processed."""

_doc_param_log = \
    """A file-like object (e.g., `sys.stderr`) to print progress information."""
