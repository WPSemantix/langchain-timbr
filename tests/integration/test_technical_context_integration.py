"""Integration tests for technical context with real ontology (timbr_crunchbase_llm_tests)."""

import pytest

from langchain_timbr import ExecuteTimbrQueryChain
from langchain_timbr.technical_context import build_technical_context
from langchain_timbr.technical_context.config import TechnicalContextConfig
from langchain_timbr.technical_context.statistics_loader import load_column_statistics
from langchain_timbr.technical_context.statistics_loader.config import StatisticsLoaderConfig


ONTOLOGY = "timbr_crunchbase_llm_tests"


class TestTechnicalContextStatistics:
    """Test that statistics are correctly loaded from the crunchbase ontology."""

    def _conn_params(self, config):
        return {
            "url": config["timbr_url"],
            "token": config["timbr_token"],
            "ontology": ONTOLOGY,
            "verify_ssl": config["verify_ssl"],
        }

    def test_organization_region_has_expected_values(self, config):
        """Property 'region' on concept 'organization' should contain Argentina - Other and Salt Lake City."""
        conn_params = self._conn_params(config)
        columns = [{"name": "region", "type": "varchar"}]

        stats_map = load_column_statistics(
            schema="dtimbr",
            table_name="organization",
            columns=columns,
            conn_params=conn_params,
        )

        assert "region" in stats_map, "Stats should be loaded for 'region'"
        stats = stats_map["region"]
        assert stats.top_k, "region should have top_k values"

        values = [str(e.value) for e in stats.top_k]
        assert "Argentina - Other" in values, (
            f"'Argentina - Other' should be in region top_k values, got: {values[:20]}"
        )
        assert "Salt Lake City" in values, (
            f"'Salt Lake City' should be in region top_k values, got: {values[:20]}"
        )

    def test_person_degree_type_has_phd_candidate(self, config):
        """Relationship property has_degree[degree].degree_type on person should contain 'Ph.D. Candidate'."""
        conn_params = self._conn_params(config)
        columns = [{"name": "has_degree[degree].degree_type", "type": "varchar"}]

        stats_map = load_column_statistics(
            schema="dtimbr",
            table_name="person",
            columns=columns,
            conn_params=conn_params,
        )

        assert "has_degree[degree].degree_type" in stats_map, (
            "Stats should be loaded for 'has_degree[degree].degree_type'"
        )
        stats = stats_map["has_degree[degree].degree_type"]
        assert stats.top_k, "has_degree[degree].degree_type should have top_k values"

        values = [str(e.value) for e in stats.top_k]
        assert "Ph.D. Candidate" in values, (
            f"'Ph.D. Candidate' should be in degree_type top_k values, got: {values[:20]}"
        )

    def test_technical_context_properties_whitelist(self, config):
        """Only properties in technical_context_properties should have stats returned."""
        conn_params = self._conn_params(config)
        columns = [
            {"name": "country_code", "type": "varchar"},
            {"name": "region", "type": "varchar"},
            {"name": "status", "type": "varchar"},
        ]

        stats_config = StatisticsLoaderConfig(include_properties=["region"])
        stats_map = load_column_statistics(
            schema="dtimbr",
            table_name="organization",
            columns=columns,
            conn_params=conn_params,
            config=stats_config,
        )

        assert "region" in stats_map, "region should have stats when whitelisted"
        assert stats_map["region"].top_k, "region should have top_k values"
        # country_code and status should NOT have stats (not in whitelist)
        assert "country_code" not in stats_map or stats_map["country_code"].top_k is None or len(stats_map["country_code"].top_k) == 0, (
            "country_code should not have stats when not in whitelist"
        )
        assert "status" not in stats_map or stats_map["status"].top_k is None or len(stats_map["status"].top_k) == 0, (
            "status should not have stats when not in whitelist"
        )

    def test_exclude_properties_blacklist(self, config):
        """Properties in exclude_properties should not have stats returned."""
        conn_params = self._conn_params(config)
        columns = [
            {"name": "country_code", "type": "varchar"},
            {"name": "region", "type": "varchar"},
            {"name": "status", "type": "varchar"},
        ]

        stats_config = StatisticsLoaderConfig(exclude_properties=["region"])
        stats_map = load_column_statistics(
            schema="dtimbr",
            table_name="organization",
            columns=columns,
            conn_params=conn_params,
            config=stats_config,
        )

        # region should NOT have stats (blacklisted)
        assert "region" not in stats_map or stats_map["region"].top_k is None or len(stats_map["region"].top_k) == 0, (
            "region should not have stats when blacklisted"
        )
        # country_code and status should still have stats
        assert "country_code" in stats_map, "country_code should have stats when not blacklisted"
        assert stats_map["country_code"].top_k, "country_code should have top_k values"
        assert "status" in stats_map, "status should have stats when not blacklisted"
        assert stats_map["status"].top_k, "status should have top_k values"


class TestTechnicalContextModes:
    """Test build_technical_context with all modes on the crunchbase ontology."""

    def _conn_params(self, config):
        return {
            "url": config["timbr_url"],
            "token": config["timbr_token"],
            "ontology": ONTOLOGY,
            "verify_ssl": config["verify_ssl"],
        }

    @pytest.mark.parametrize("mode", ["include_all", "filter_matched", "auto"])
    def test_build_tc_organization_region(self, llm, config, mode):
        """build_technical_context should annotate 'region' for organization across all modes."""
        conn_params = self._conn_params(config)
        columns = [
            {"name": "region", "type": "varchar"},
            {"name": "name", "type": "varchar"},
            {"name": "status", "type": "varchar"},
        ]
        tc_config = TechnicalContextConfig(mode=mode, max_tokens=3000)

        result = build_technical_context(
            question="Count companies in Salt Lake",
            columns=columns,
            schema="dtimbr",
            concept="organization",
            conn_params=conn_params,
            config=tc_config,
            llm=llm,
        )

        assert result.column_annotations is not None
        assert result.metadata["effective_mode"] in ("include_all", "filter_matched")
        # region must be annotated because it has values relevant to the question
        assert "region" in result.column_annotations, (
            f"'region' should be annotated in mode={mode}, annotations: {list(result.column_annotations.keys())}"
        )
        assert "Salt Lake City" in result.column_annotations["region"]

    @pytest.mark.parametrize("mode", ["include_all", "filter_matched", "auto"])
    def test_build_tc_person_degree(self, llm, config, mode):
        """build_technical_context should annotate degree_type for person across all modes."""
        conn_params = self._conn_params(config)
        columns = [
            {"name": "has_degree[degree].degree_type", "type": "varchar"},
            {"name": "first_name", "type": "varchar"},
            {"name": "last_name", "type": "varchar"},
        ]
        tc_config = TechnicalContextConfig(mode=mode, max_tokens=3000)

        result = build_technical_context(
            question="Count person that are PHD candidate",
            columns=columns,
            schema="dtimbr",
            concept="person",
            conn_params=conn_params,
            config=tc_config,
            llm=llm,
        )

        assert result.column_annotations is not None
        assert result.metadata["effective_mode"] in ("include_all", "filter_matched")
        assert "has_degree[degree].degree_type" in result.column_annotations, (
            f"'has_degree[degree].degree_type' should be annotated in mode={mode}, "
            f"annotations: {list(result.column_annotations.keys())}"
        )
        assert "Ph.D. Candidate" in result.column_annotations["has_degree[degree].degree_type"]


class TestTechnicalContextQueryExecution:
    """End-to-end tests: TC-enriched SQL generation produces correct queries."""

    @pytest.mark.parametrize("mode", ["include_all", "filter_matched", "auto"])
    def test_salt_lake_city_filter(self, llm, config, mode):
        """'Count companies in Salt Lake' should produce SQL filtering on region with 'Salt Lake City'."""
        chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=ONTOLOGY,
            concepts_list="organization",
            views_list="None",
            verify_ssl=config["verify_ssl"],
            enable_technical_context=True,
            technical_context_mode=mode,
            technical_context_max_tokens=3000,
        )

        result = chain.invoke({"prompt": "Count companies in Salt Lake"})
        print(f"[mode={mode}] Salt Lake result:", result)

        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"
        assert result["sql"], "SQL should be present"
        assert "Salt Lake City" in result["sql"], (
            f"SQL should filter on 'Salt Lake City', got: {result['sql']}"
        )
        assert "%" not in result["sql"], (
            f"SQL should filter on 'Salt Lake City' not like %, got: {result['sql']}"
        )
        assert "like" not in result["sql"].lower(), (
            f"SQL should filter on 'Salt Lake City' not like %, got: {result['sql']}"
        )

    @pytest.mark.parametrize("mode", ["include_all", "filter_matched", "auto"])
    def test_argentina_operating_filter(self, llm, config, mode):
        """'Count companies in Argentina and operate' should filter on region and status."""
        
        chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=ONTOLOGY,
            concepts_list="organization",
            exclude_properties=["country_code"],
            views_list="None",
            verify_ssl=config["verify_ssl"],
            enable_technical_context=True,
            technical_context_mode=mode,
            technical_context_max_tokens=3000
        )

        result = chain.invoke({"prompt": "Count companies in Argentina region and operate"})
        print(f"[mode={mode}] Argentina operating result:", result)

        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"
        assert result["sql"], "SQL should be present"
        sql_lower = result["sql"].lower()
        assert "argentina" in sql_lower, (
            f"SQL should filter on Argentina, got: {result['sql']}"
        )
        assert "operating" in sql_lower, (
            f"SQL should filter on operating status, got: {result['sql']}"
        )

    @pytest.mark.parametrize("mode", ["include_all", "filter_matched", "auto"])
    def test_phd_candidate_filter(self, llm, config, mode):
        """'Count person that are PHD candidate' should filter on degree_type."""
        chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=ONTOLOGY,
            concepts_list="person",
            views_list="None",
            verify_ssl=config["verify_ssl"],
            enable_technical_context=True,
            technical_context_mode=mode,
            technical_context_max_tokens=3000,
        )

        result = chain.invoke({"prompt": "Count person that are PHD candidate"})
        print(f"[mode={mode}] PhD candidate result:", result)

        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"
        assert result["sql"], "SQL should be present"
        assert "Ph.D. Candidate" in result["sql"], (
            f"SQL should filter on 'Ph.D. Candidate', got: {result['sql']}"
        )
    @pytest.mark.parametrize("mode", ["include_all", "filter_matched", "auto"])
    def test_cube_region_filter(self, llm, config, mode):
        """'Count companies in San Francisco Bay Area that are not active anymore"""
        
        chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=ONTOLOGY,
            concepts_list="organization",
            exclude_properties=["country_code"],
            views_list="None",
            verify_ssl=config["verify_ssl"],
            enable_technical_context=True,
            technical_context_mode=mode,
            technical_context_max_tokens=3000
        )

        result = chain.invoke({"prompt": "Count companies in San Francisco Bay Area that are not active anymore."})
        print(f"[mode={mode}] San Francisco Bay Area not operating result:", result)

        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"
        assert result["sql"], "SQL should be present"
        sql_lower = result["sql"].lower()
        assert "sf bay" in sql_lower, (
            f"SQL should filter on SF Bay Area, got: {result['sql']}"
        )

        assert "%" not in sql_lower, (
            f"SQL should filter on SF Bay Area, got: {result['sql']}"
        )

        assert "closed" in sql_lower or "!= 'operating'" in sql_lower, (
            f"SQL should filter on operating status, got: {result['sql']}"
        )   