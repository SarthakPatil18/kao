const hre = require("hardhat");

async function main() {
  console.log("Deploying AttestationRegistry...");

  const AttestationRegistry = await hre.ethers.getContractFactory(
    "AttestationRegistry"
  );
  const registry = await AttestationRegistry.deploy();
  await registry.waitForDeployment();

  const address = await registry.getAddress();
  console.log(`AttestationRegistry deployed to: ${address}`);
  console.log(
    `Block explorer: https://amoy.polygonscan.com/address/${address}`
  );
  console.log("");
  console.log("Add this to your .env file:");
  console.log(`CONTRACT_ADDRESS=${address}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
