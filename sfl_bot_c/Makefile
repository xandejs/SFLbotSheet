CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude -Ixgboost_include
LDFLAGS = -lcurl -lcjson -lssl -lcrypto -lm

# Aponte XGB_LIB_DIR para a pasta que tem libxgboost.so — por padrão, a
# instalada junto com o pacote pip `xgboost` (usada só para linkar; em
# produção real seria melhor instalar a lib do sistema ou buildar da fonte).
XGB_LIB_DIR ?= $(shell python3 -c "import xgboost,os;print(os.path.join(os.path.dirname(xgboost.__file__),'lib'))" 2>/dev/null)

SRC = src/util.c src/oauth.c src/sheets.c src/indicators.c src/calibration.c src/xgb_wrapper.c src/main.c
OBJ = $(SRC:src/%.c=build/%.o)
BIN = build/sfl_bot

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(OBJ) -o $(BIN) -L$(XGB_LIB_DIR) -lxgboost -Wl,-rpath,$(XGB_LIB_DIR) $(LDFLAGS)

build/%.o: src/%.c
	@mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf build

.PHONY: all clean
