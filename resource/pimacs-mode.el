;;; pimacs-mode.el --- Major mode for editing Pimacs code -*- lexical-binding: t; -*-

(defvar pimacs-mode-syntax-table
  (let ((syntax-table (make-syntax-table)))
    ;; Dot and underscore are considered word characters
    (modify-syntax-entry ?. "w" syntax-table)
    (modify-syntax-entry ?_ "w" syntax-table)
    ;; Percent sign is also considered a word character
    (modify-syntax-entry ?% "w" syntax-table)
    ;; Parentheses are paired
    (modify-syntax-entry ?\( "()" syntax-table)
    (modify-syntax-entry ?\) ")(" syntax-table)
    ;; Square brackets are paired
    (modify-syntax-entry ?\[ "(]" syntax-table)
    (modify-syntax-entry ?\] ")[" syntax-table)
    syntax-table)
  "Syntax table for `pimacs-mode`.")

(setq pimacs-keywords
      '("def" "class" "var" "let" "return" "if" "else" "while" "for" "in"))
(setq pimacs-builtin-type-keywords
      '("Int" "Float" "Str" "Dict" "List" "Tuple" "Any"))

(defvar pimacs-font-lock-keywords
  (list
   ;; Keywords
   ;; `(,(rx-to-string '(and bol (or "def" "class" "var" "let" "return" "if" "else" "while" "for" "in") (not (any word "(" "["))))
   ;;   0 font-lock-keyword-face)
   `(,(regexp-opt pimacs-keywords 'words) . font-lock-keyword-face)

   ;; Types
   `(,(regexp-opt pimacs-builtin-type-keywords 'words) . font-lock-type-face)
   ;; Constants/functions prefixed with %
   `(,(rx-to-string '(and bol "%" (not (any word "(" "["))))
     0 font-lock-constant-face)
   ;; Function names
   '("\\<\\([_A-Za-z-%][A-Za-z0-9_-]*\\)(" 1 font-lock-function-name-face)
   ;; Variables and parameters
   '("\\<\\([A-Za-z-][A-Za-z0-9_-]*\\):" 1 font-lock-variable-name-face)
   ;; Comments
   '("#\\(?:.\\|\n\\)*?$" 0 font-lock-comment-face))
  "Font-lock keywords for `pimacs-mode`.")

(defun pimacs-indent-line ()
  "Indent the current line for `pimacs-mode`."
  (interactive)
  (let ((indent-level 0))
    (save-excursion
      (beginning-of-line)
      (skip-chars-forward " \t")
      (cond
       ((looking-at-p "def[ \t]*\\([^)]+\\)") (setq indent-level 4))
       ((looking-at-p "class[ \t]*\\([^)]+\\)") (setq indent-level 4))
       ((looking-at-p "if[ \t]*\\([^)]+\\)") (setq indent-level 4))
       ((looking-at-p "else[ \t]*\\([^)]+\\)") (setq indent-level 4))
       ((looking-at-p "while[ \t]*\\([^)]+\\)") (setq indent-level 4))
       ((looking-at-p "for[ \t]*\\([^)]+\\)") (setq indent-level 4))
       (t (setq indent-level 2))))
    (indent-line-to indent-level)))

(define-derived-mode pimacs-mode prog-mode "Pimacs"
  "Major mode for editing Pimacs code."
  ;; Initialize syntax table
  (set-syntax-table pimacs-mode-syntax-table)
  ;; Set font-lock keywords
  (setq font-lock-defaults '(pimacs-font-lock-keywords))
  ;; Set indentation function
  (setq-local indent-line-function 'pimacs-indent-line))

;; Add file extension associations
(add-to-list 'auto-mode-alist '("\\.pim$" . pimacs-mode))
(add-to-list 'auto-mode-alist '("\\.pimacs$" . pimacs-mode))

(provide 'pimacs-mode)

;;; pimacs-mode.el ends here
