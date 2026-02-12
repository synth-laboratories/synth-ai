use once_cell::sync::Lazy;
use serde_json::{Map, Value};

static LEVER_SENSOR_V1_CONTRACT: Lazy<Value> = Lazy::new(|| {
    let raw = include_str!("../../assets/lever_sensor_v1_schema.json");
    serde_json::from_str(raw).unwrap_or_else(|_| Value::Object(Map::new()))
});

/// Return the frozen v1 Levers/Sensors contract schema.
pub fn lever_sensor_v1_contract_schema() -> Value {
    LEVER_SENSOR_V1_CONTRACT.clone()
}

#[cfg(test)]
mod tests {
    use super::lever_sensor_v1_contract_schema;

    #[test]
    fn contract_has_required_top_level_keys() {
        let schema = lever_sensor_v1_contract_schema();
        let required = schema
            .get("required")
            .and_then(|v| v.as_array())
            .expect("required must be an array");

        for key in [
            "scope_key_order",
            "required_lever_kinds",
            "required_sensor_kinds",
            "sensor_frame_minimum",
        ] {
            assert!(
                required.iter().any(|item| item.as_str() == Some(key)),
                "missing required key: {key}"
            );
        }
    }

    #[test]
    fn contract_scope_key_order_matches_v1() {
        let schema = lever_sensor_v1_contract_schema();
        let order = schema
            .get("properties")
            .and_then(|v| v.get("scope_key_order"))
            .and_then(|v| v.get("items"))
            .and_then(|v| v.get("enum"))
            .and_then(|v| v.as_array())
            .expect("scope_key_order enum must be present");

        let expected = [
            "org",
            "project",
            "horizon",
            "job",
            "stage",
            "seed",
            "rollout",
            "graph_node",
            "tool_call",
            "user",
        ];

        assert_eq!(order.len(), expected.len());
        for (actual, wanted) in order.iter().zip(expected.iter()) {
            assert_eq!(actual.as_str(), Some(*wanted));
        }
    }

    #[test]
    fn contract_required_kinds_match_v1() {
        let schema = lever_sensor_v1_contract_schema();

        let lever_kinds = schema
            .get("properties")
            .and_then(|v| v.get("required_lever_kinds"))
            .and_then(|v| v.get("items"))
            .and_then(|v| v.get("enum"))
            .and_then(|v| v.as_array())
            .expect("required_lever_kinds enum must be present");
        let expected_levers = [
            "prompt",
            "context",
            "code",
            "constraint",
            "note",
            "spec",
            "graph_yaml",
            "variable",
            "experiment",
        ];
        assert_eq!(lever_kinds.len(), expected_levers.len());
        for expected in expected_levers {
            assert!(lever_kinds
                .iter()
                .any(|item| item.as_str() == Some(expected)));
        }

        let sensor_kinds = schema
            .get("properties")
            .and_then(|v| v.get("required_sensor_kinds"))
            .and_then(|v| v.get("items"))
            .and_then(|v| v.get("enum"))
            .and_then(|v| v.as_array())
            .expect("required_sensor_kinds enum must be present");
        let expected_sensors = [
            "reward",
            "timing",
            "rollout",
            "resource",
            "safety",
            "quality",
            "trace",
            "context_apply",
            "experiment",
        ];
        assert_eq!(sensor_kinds.len(), expected_sensors.len());
        for expected in expected_sensors {
            assert!(sensor_kinds
                .iter()
                .any(|item| item.as_str() == Some(expected)));
        }
    }

    #[test]
    fn contract_sensor_frame_minimum_fields_match_v1() {
        let schema = lever_sensor_v1_contract_schema();
        let sensor_frame = schema
            .get("properties")
            .and_then(|v| v.get("sensor_frame_minimum"))
            .expect("sensor_frame_minimum must be present");

        let required_fields = sensor_frame
            .get("properties")
            .and_then(|v| v.get("required_fields"))
            .and_then(|v| v.get("items"))
            .and_then(|v| v.get("enum"))
            .and_then(|v| v.as_array())
            .expect("sensor_frame_minimum.required_fields enum must be present");
        let expected_frame_fields = [
            "scope",
            "sensors",
            "lever_versions",
            "trace_ids",
            "created_at",
        ];
        assert_eq!(required_fields.len(), expected_frame_fields.len());
        for expected in expected_frame_fields {
            assert!(required_fields
                .iter()
                .any(|item| item.as_str() == Some(expected)));
        }

        let required_sensor_fields = sensor_frame
            .get("properties")
            .and_then(|v| v.get("required_sensor_fields"))
            .and_then(|v| v.get("items"))
            .and_then(|v| v.get("enum"))
            .and_then(|v| v.as_array())
            .expect("sensor_frame_minimum.required_sensor_fields enum must be present");
        let expected_sensor_fields = ["sensor_id", "kind", "scope", "value", "timestamp"];
        assert_eq!(required_sensor_fields.len(), expected_sensor_fields.len());
        for expected in expected_sensor_fields {
            assert!(required_sensor_fields
                .iter()
                .any(|item| item.as_str() == Some(expected)));
        }
    }
}
