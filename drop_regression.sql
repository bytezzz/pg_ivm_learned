DROP DATABASE regression;

DROP TABLESPACE regress_tblspace;
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN SELECT rolname FROM pg_roles WHERE rolname LIKE 'regress%'
    LOOP
        EXECUTE 'DROP ROLE ' || quote_ident(r.rolname);
    END LOOP;
END $$;
