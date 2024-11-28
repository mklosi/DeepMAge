#!/bin/bash
set -e

## errors: 
# GSE128235 - ValueError: Not an IDAT file. Unsupported file type.
# GSE105123 - FileNotFoundError: [Errno 2] No such file or directory: 'resources_methylprep/GSE105123_download_pandas_1.3.5/GSE105123_family.xml'
# GSE106648 - ERROR:methylprep.download.process_data:[!] Geo data set GSE106648 probably does NOT contain usable raw data (in .idat format). Not downloading.
# GSE77696 - ERROR:methylprep.download.process_data:[!] Geo data set GSE77696 probably does NOT contain usable raw data (in .idat format). Not downloading.

for gse_id in GSE59065 GSE106648 GSE61496 GSE107459 GSE77696
do
    echo "--- $gse_id ------------------"
    python -m methylprep -v download -i $gse_id -d resources_methylprep/"$gse_id"_download_pandas_1.3.5
    echo "AAA done with download"
    rm -rf resources_methylprep/"$gse_id"_download_pandas_1.3.5/*/*.idat
    echo "AAA done with delete"
done
