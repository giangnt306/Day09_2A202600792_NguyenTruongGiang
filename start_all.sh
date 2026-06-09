#!/bin/bash
set -e

# Start all Legal Multi-Agent System services
# Registry must be first, then leaf agents, then orchestrators

echo "Starting Registry service on port 10000..."
python -m registry &
REGISTRY_PID=$!
sleep 2

echo "Starting Tax Agent on port 10102..."
python -m tax_agent &
TAX_PID=$!

echo "Starting Compliance Agent on port 10103..."
python -m compliance_agent &
COMPLIANCE_PID=$!
sleep 3

echo "Starting Law Agent on port 10101..."
python -m law_agent &
LAW_PID=$!
sleep 3

echo "Starting Customer Agent on port 10100..."
python -m customer_agent &
CUSTOMER_PID=$!

echo ""
echo "All services started:"
echo "  Registry:         http://localhost:10000"
echo "  Customer Agent:   http://localhost:10100"
echo "  Law Agent:        http://localhost:10101"
echo "  Tax Agent:        http://localhost:10102"
echo "  Compliance Agent: http://localhost:10103"
echo ""
echo "Run test_client.py to send a query:"
echo "  python test_client.py"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait for all background processes
wait $REGISTRY_PID $TAX_PID $COMPLIANCE_PID $LAW_PID $CUSTOMER_PID