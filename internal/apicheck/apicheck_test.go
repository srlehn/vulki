package apicheck_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

const modulePath = "github.com/srlehn/vulki"

var publicPackageDirs = []string{
	".",
	"registration",
	"shader",
	"vk",
}

func TestPublicAPIDoesNotExposeThirdPartyTypes(t *testing.T) {
	root := repositoryRoot(t)
	var violations []string
	for _, relativeDir := range publicPackageDirs {
		dir := filepath.Join(root, relativeDir)
		entries, err := os.ReadDir(dir)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			t.Fatalf("read public package %s: %v", relativeDir, err)
		}

		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") ||
				strings.HasSuffix(entry.Name(), "_test.go") {
				continue
			}
			filename := filepath.Join(dir, entry.Name())
			violations = append(violations, inspectPublicFile(t, filename)...)
		}
	}

	if len(violations) == 0 {
		return
	}
	sort.Strings(violations)
	for _, violation := range violations {
		t.Error(violation)
	}
}

func TestPublicAPICheckRejectsThirdPartyExport(t *testing.T) {
	dir := t.TempDir()
	filename := filepath.Join(dir, "leak.go")
	source := []byte(`package leak

import dependency "example.com/thirdparty/dependency"

type Public struct {
	Leaked dependency.Value
	hidden dependency.Value
}
`)
	if err := os.WriteFile(filename, source, 0o600); err != nil {
		t.Fatalf("write synthetic package: %v", err)
	}

	violations := inspectPublicFile(t, filename)
	if got, want := len(violations), 1; got != want {
		t.Fatalf("violations = %v, want %d exported-field violation", violations, want)
	}
	if !strings.Contains(violations[0], `dependency.Value from "example.com/thirdparty/dependency"`) {
		t.Fatalf("violation = %q, want dependency path and type", violations[0])
	}
}

func repositoryRoot(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate API check source")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "..", ".."))
}

func inspectPublicFile(t *testing.T, filename string) []string {
	t.Helper()
	files := token.NewFileSet()
	parsed, err := parser.ParseFile(files, filename, nil, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse %s: %v", filename, err)
	}

	externalImports := make(map[string]string)
	var violations []string
	for _, imported := range parsed.Imports {
		path, err := strconv.Unquote(imported.Path.Value)
		if err != nil {
			t.Fatalf("unquote import in %s: %v", filename, err)
		}
		if allowedImport(path) {
			continue
		}

		name := filepath.Base(path)
		if imported.Name != nil {
			name = imported.Name.Name
		}
		switch name {
		case "_":
			continue
		case ".":
			position := files.Position(imported.Pos())
			violations = append(violations, fmt.Sprintf(
				"%s: public package dot-imports third-party package %q",
				position, path,
			))
			continue
		}
		externalImports[name] = path
	}

	checkExpr := func(expr ast.Expr) {
		ast.Inspect(expr, func(node ast.Node) bool {
			selector, ok := node.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			identifier, ok := selector.X.(*ast.Ident)
			if !ok {
				return true
			}
			path, ok := externalImports[identifier.Name]
			if !ok {
				return true
			}
			position := files.Position(selector.Pos())
			violations = append(violations, fmt.Sprintf(
				"%s: exported API references %s.%s from %q",
				position, identifier.Name, selector.Sel.Name, path,
			))
			return true
		})
	}

	checkFields := func(fields *ast.FieldList, exportedOnly bool) {
		if fields == nil {
			return
		}
		for _, field := range fields.List {
			if exportedOnly && len(field.Names) > 0 {
				exported := false
				for _, name := range field.Names {
					if name.IsExported() {
						exported = true
						break
					}
				}
				if !exported {
					continue
				}
			}
			checkExpr(field.Type)
		}
	}

	for _, declaration := range parsed.Decls {
		switch declaration := declaration.(type) {
		case *ast.FuncDecl:
			if !declaration.Name.IsExported() || !exportedReceiver(declaration.Recv) {
				continue
			}
			checkFields(declaration.Type.TypeParams, false)
			checkFields(declaration.Type.Params, false)
			checkFields(declaration.Type.Results, false)

		case *ast.GenDecl:
			for _, specification := range declaration.Specs {
				switch specification := specification.(type) {
				case *ast.TypeSpec:
					if !specification.Name.IsExported() {
						continue
					}
					checkFields(specification.TypeParams, false)
					switch typeExpr := specification.Type.(type) {
					case *ast.StructType:
						checkFields(typeExpr.Fields, true)
					case *ast.InterfaceType:
						checkFields(typeExpr.Methods, false)
					default:
						checkExpr(typeExpr)
					}

				case *ast.ValueSpec:
					exported := false
					for _, name := range specification.Names {
						if name.IsExported() {
							exported = true
							break
						}
					}
					if !exported {
						continue
					}
					if specification.Type != nil {
						checkExpr(specification.Type)
					}
					for _, value := range specification.Values {
						checkExpr(value)
					}
				}
			}
		}
	}
	return violations
}

func exportedReceiver(receivers *ast.FieldList) bool {
	if receivers == nil {
		return true
	}
	if len(receivers.List) != 1 {
		return false
	}
	expr := receivers.List[0].Type
	for {
		switch typed := expr.(type) {
		case *ast.StarExpr:
			expr = typed.X
		case *ast.IndexExpr:
			expr = typed.X
		case *ast.IndexListExpr:
			expr = typed.X
		case *ast.Ident:
			return typed.IsExported()
		default:
			return false
		}
	}
}

func allowedImport(path string) bool {
	if path == modulePath || strings.HasPrefix(path, modulePath+"/") {
		return true
	}
	first, _, _ := strings.Cut(path, "/")
	return !strings.Contains(first, ".")
}
