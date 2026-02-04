# Prompt
Analyze the given pdf file and extract data found in tables with either of these columns present containing: "Type, Vendor, Description/Objective, Owner, PAG, Progress to Date, Target Completion." or
"Use Case, Update/Next Steps"

These tables are not contiguose in the document, collect data found in the tables to produce a full set of rows.

Extract text from these tables according to the provided schema. In the source tables columns, "Type"  and "Use Case" can be assigned to the "project_type" attribute in the schema. "Updates/Next Steps" and "Progress to Date" can be assigned  to the "progress_to_date" attribute in the schema.


# Schema
```schema
{
  "type": "json_schema",
  "json_schema": {
    "name": "extraction",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
         "tables": {
            "type": "array",
            "items" : {
                "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "page": { "type": "integer"},
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "project_type": {"type": "string"},
                                    "vendor": {"type": "string"},
                                    "description": {"type": "string"},
                                    "owner": {"type": "string"},
                                    "pag": {"type": "string"},
                                    "progress_to_date": {"type": "string"},
                                    "target_completion": {"type": "string"}
                                },
                                "required": ["project_type"],
                                "additionalProperties": false
                            }
                        }
                    }
                }
            }
        }
    }
  }
}
```
