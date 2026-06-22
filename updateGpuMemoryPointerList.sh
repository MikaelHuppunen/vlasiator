#!/usr/bin/env bash
# Collects all pointers passed to gpuMemoryManager
# and writes them into ./arch/gpu_memory_manager.hpp between
# //DEFINITIONS_HERE and //DEFINITIONS_END markers.

shopt -s globstar nullglob

declare -A seen

# --- Collect unique identifiers ---
for file in **/*.cpp **/*.h **/*.hpp; do
   # Skip any files inside directories named "libraries" or "submodules"
   [[ "$file" == libraries* ]] && continue
   [[ "$file" == */libraries* ]] && continue
   [[ "$file" == submodules* ]] && continue
   [[ "$file" == */submodules* ]] && continue

   while IFS= read -r match; do
      [[ -z "$match" ]] && continue
      seen["$match"]=1
   done < <(
      # Match CREATE_UNIQUE_POINTER(obj, identifier)
      grep -oP 'CREATE_UNIQUE_POINTER\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*\)' "$file" 2>/dev/null |
      sed -E 's/.*CREATE_UNIQUE_POINTER\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\).*/\1/'

      # Match CREATE_SUBPOINTERS(obj, identifier, ...)
      grep -oP 'CREATE_SUBPOINTERS\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*,[^)]*\)' "$file" 2>/dev/null |
      sed -E 's/.*CREATE_SUBPOINTERS\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*,.*/\1/'

      grep -oE 'SESSION(_HOST)?_ALLOCATE[[:space:]]*\([[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*,[[:space:]]*[A-Za-z_][A-Za-z0-9_]*' "$file" |
      sed -E 's/.*,\s*([A-Za-z_][A-Za-z0-9_]*)$/\1/'

   )
done


# --- Prepare the definitions content ---
definitions=$(for key in "${!seen[@]}"; do echo "$key"; done | sort)

# --- Update gpu_memory_manager.hpp ---
target="./arch/gpu_memory_manager.hpp"

if [[ ! -f "$target" ]]; then
   echo "[ERROR] $target not found."
   exit 1
fi

# Create a backup
cp "$target" "$target.bak"

# Use awk to replace content between #define DEFINITIONS_HERE / END markers
# Generate new content in a temp file
tmpfile=$(mktemp)

awk -v defs="$definitions" '
   BEGIN { split(defs, arr, "\n") }
   /^[[:space:]]*#define[[:space:]]+DEFINITIONS_HERE/ {
      print;  # print the marker exactly as it appears
      first = 1
      line = "   size_t "
      for (i in arr) {
         if (length(arr[i]) > 0) {
            if (!first) line = line ", "
            line = line arr[i]
            first = 0
         }
      }
      if (!first) print line ";"
      skip = 1
      next
   }
   /^[[:space:]]*#define[[:space:]]+DEFINITIONS_END/ {
      skip = 0
   }
   !skip
' "$target.bak" > "$tmpfile"

# Only overwrite if the content actually changed
if ! cmp -s "$tmpfile" "$target"; then
   mv "$tmpfile" "$target"
   echo "[SCAN] Updated $target"
else
   rm "$tmpfile"
fi

rm "$target.bak"
