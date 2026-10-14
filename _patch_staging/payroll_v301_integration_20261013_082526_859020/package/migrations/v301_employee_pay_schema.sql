CREATE TABLE IF NOT EXISTS employee_pay_profiles (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE,
    work_state VARCHAR(2) NOT NULL DEFAULT 'CA',
    pay_type VARCHAR(10) NOT NULL,
    exemption_status VARCHAR(15) NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    FOREIGN KEY(user_id) REFERENCES user(id)
);

CREATE TABLE IF NOT EXISTS employee_pay_rate_history (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER NOT NULL,
    effective_from DATE NOT NULL,
    effective_to DATE,
    amount NUMERIC(14,2) NOT NULL,
    pay_type VARCHAR(10) NOT NULL,
    exemption_status VARCHAR(15) NOT NULL,
    work_state VARCHAR(2) NOT NULL,
    created_by_id INTEGER NOT NULL,
    created_at DATETIME NOT NULL,
    CHECK(amount >= 0),
    CHECK(effective_to IS NULL OR effective_to >= effective_from),
    FOREIGN KEY(profile_id) REFERENCES employee_pay_profiles(id),
    FOREIGN KEY(created_by_id) REFERENCES user(id)
);

CREATE INDEX IF NOT EXISTS ix_employee_pay_rate_profile
ON employee_pay_rate_history(profile_id);

CREATE INDEX IF NOT EXISTS ix_employee_pay_rate_effective
ON employee_pay_rate_history(profile_id, effective_from);

CREATE TABLE IF NOT EXISTS employee_pay_audit (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    actor_id INTEGER NOT NULL,
    action VARCHAR(50) NOT NULL,
    old_values TEXT,
    new_values TEXT NOT NULL,
    created_at DATETIME NOT NULL,
    FOREIGN KEY(user_id) REFERENCES user(id),
    FOREIGN KEY(actor_id) REFERENCES user(id)
);

CREATE INDEX IF NOT EXISTS ix_employee_pay_audit_user
ON employee_pay_audit(user_id, created_at);
