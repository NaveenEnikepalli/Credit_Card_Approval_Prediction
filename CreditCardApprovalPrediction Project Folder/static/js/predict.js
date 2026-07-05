/**
 * Prediction Form Frontend Logic
 * Includes dynamic field validations, error indicators, submit loading state, and reset confirmations.
 */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resetBtn = document.getElementById('btn-reset');
    const predictBtn = document.getElementById('btn-predict');
    const btnSpinner = predictBtn.querySelector('.btn-spinner');
    const btnText = predictBtn.querySelector('.btn-text');
    
    // Select all inputs and selects
    const fields = form.querySelectorAll('input, select');
    
    // Helper: Show error on UI
    const showError = (field, message) => {
        const formField = field.closest('.form-field');
        if (formField) {
            formField.classList.add('invalid');
            const errorSpan = formField.querySelector('.error-msg');
            if (errorSpan) {
                errorSpan.textContent = message;
            }
        }
    };

    // Helper: Clear error from UI
    const clearError = (field) => {
        const formField = field.closest('.form-field');
        if (formField) {
            formField.classList.remove('invalid');
            const errorSpan = formField.querySelector('.error-msg');
            if (errorSpan) {
                errorSpan.textContent = '';
            }
        }
    };

    // Main Field Validation Function
    const validateField = (field) => {
        const id = field.id;
        const val = field.value.trim();
        
        // 1. Required Check
        if (val === '') {
            showError(field, 'This field is required.');
            return false;
        }

        // 2. Individual Type & Range Validations
        if (field.type === 'number') {
            const num = parseFloat(val);
            
            if (isNaN(num)) {
                showError(field, 'Please enter a valid numeric value.');
                return false;
            }

            // Age: 18 - 100
            if (id === 'age_years') {
                if (!Number.isInteger(Number(val)) || num < 18 || num > 100) {
                    showError(field, 'Age must be an integer between 18 and 100.');
                    return false;
                }
            }

            // Employment Years: 0 - 60
            if (id === 'employment_years') {
                if (num < 0 || num > 60) {
                    showError(field, 'Employment duration must be between 0 and 60 years.');
                    return false;
                }
            }

            // Annual Income: > 0
            if (id === 'amt_income_total') {
                if (num <= 0) {
                    showError(field, 'Annual income must be greater than zero.');
                    return false;
                }
            }

            // Credit Limit: > 0
            if (id === 'amt_credit') {
                if (num <= 0) {
                    showError(field, 'Credit limit must be greater than zero.');
                    return false;
                }
            }

            // Loan Annuity: > 0
            if (id === 'amt_annuity') {
                if (num <= 0) {
                    showError(field, 'Annuity must be greater than zero.');
                    return false;
                }
            }

            // Children: 0 - 20
            if (id === 'cnt_children') {
                if (!Number.isInteger(Number(val)) || num < 0 || num > 20) {
                    showError(field, 'Children count must be an integer between 0 and 20.');
                    return false;
                }
            }

            // Family Size: 1 - 20
            if (id === 'cnt_fam_members') {
                if (!Number.isInteger(Number(val)) || num < 1 || num > 20) {
                    showError(field, 'Family size must be an integer between 1 and 20.');
                    return false;
                }
                
                // Extra logical check: Family members must be at least children + 1
                const childrenField = document.getElementById('cnt_children');
                const childrenVal = childrenField.value.trim();
                if (childrenVal !== '') {
                    const childrenNum = parseInt(childrenVal, 10);
                    if (!isNaN(childrenNum) && num < (childrenNum + 1)) {
                        showError(field, `Family size must be at least ${childrenNum + 1} (Children + 1 applicant).`);
                        return false;
                    }
                }
            }

            // Credit Bureau Queries: >= 0
            if (id === 'amt_req_credit_bureau_year') {
                if (!Number.isInteger(Number(val)) || num < 0) {
                    showError(field, 'Credit bureau queries must be 0 or greater.');
                    return false;
                }
            }
        }

        // Field is valid, clear error
        clearError(field);
        return true;
    };

    // Add Live Validation Listeners
    fields.forEach(field => {
        const events = ['input', 'change', 'blur'];
        events.forEach(evt => {
            field.addEventListener(evt, () => validateField(field));
        });
    });

    // Form Reset Handler
    resetBtn.addEventListener('click', () => {
        const confirmReset = confirm("Are you sure you want to reset the form? All entered details will be cleared.");
        if (confirmReset) {
            form.reset();
            // Clear all validation states
            fields.forEach(field => clearError(field));
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    });

    // Form Submit Handler
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        
        let isFormValid = true;
        
        // Validate all fields
        fields.forEach(field => {
            const isFieldValid = validateField(field);
            if (!isFieldValid) {
                isFormValid = false;
            }
        });

        if (isFormValid) {
            // Form is fully validated, trigger loading state
            predictBtn.disabled = true;
            resetBtn.disabled = true;
            btnText.textContent = "Assessing Eligibility...";
            btnSpinner.classList.remove('hidden');
            
            // Wait briefly to show spinner before posting
            setTimeout(() => {
                form.submit();
            }, 600);
        } else {
            // Scroll to the first error field
            const firstErrorField = form.querySelector('.form-field.invalid');
            if (firstErrorField) {
                const headerOffset = 100;
                const elementPosition = firstErrorField.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.scrollY - headerOffset;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
            }
        }
    });
});
