-- Run once as a database administrator after applying the DataLens migration.
-- Creating an existing role intentionally fails rather than altering its privileges.
CREATE ROLE b2b_radar_bi_reader NOLOGIN;
GRANT USAGE ON SCHEMA analytics_bi TO b2b_radar_bi_reader;
GRANT SELECT ON analytics_bi.organization_overview,
                analytics_bi.organization_monthly,
                analytics_bi.category_overview,
                analytics_bi.category_monthly TO b2b_radar_bi_reader;
-- Create a dedicated LOGIN separately, set its password interactively with psql \password,
-- then GRANT b2b_radar_bi_reader TO your_datalens_login;
-- Do not grant that LOGIN access to the analytics schema or membership in an application role.
